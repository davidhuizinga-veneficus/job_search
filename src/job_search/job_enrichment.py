"""Enrich scraped Indeed jobs using a PydanticAI Gemini agent and persist them in SQLite."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from job_search.gemini_limiter import GeminiRateLimiter, get_shared_gemini_limiter

load_dotenv()

BATCH_SIZE = 10


class Job(BaseModel):
    id: str

    source: str
    url: str
    title: str
    company: str

    location: str | None = None
    date_posted: datetime | None = None

    job_type: str | None = None
    is_remote: bool | None = None

    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_interval: str | None = None

    description: str
    search_term: str
    scraped_at: datetime

    summary: str = ""

    responsibilities: list[str] = Field(default_factory=list)

    minimum_years_experience: float | None = None
    maximum_years_experience: float | None = None

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)

    required_education: list[str] = Field(default_factory=list)
    preferred_education: list[str] = Field(default_factory=list)

    required_certifications: list[str] = Field(default_factory=list)
    preferred_certifications: list[str] = Field(default_factory=list)

    required_licenses: list[str] = Field(default_factory=list)
    preferred_licenses: list[str] = Field(default_factory=list)

    required_languages: list[str] = Field(default_factory=list)
    preferred_languages: list[str] = Field(default_factory=list)

    seniority: str | None = None
    industry: str | None = None

    benefits: list[str] = Field(default_factory=list)

    shifts: list[str] = Field(default_factory=list)
    schedule: list[str] = Field(default_factory=list)

    travel_required: bool | None = None
    travel_percentage: float | None = None

    physical_requirements: list[str] = Field(default_factory=list)
    other_requirements: list[str] = Field(default_factory=list)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def canonical_indeed_job_key(job: dict[str, Any]) -> str:
    """Normalize Indeed URLs so tracking parameters do not create duplicates."""
    raw_url = (
        str(job.get("job_url") or job.get("url") or job.get("job_url_direct") or "")
    ).strip()
    if raw_url:
        parsed = urlparse(raw_url)
        params = parse_qs(parsed.query)
        job_id = params.get("jk", [None])[0]
        if job_id:
            return f"indeed:{job_id}"

    raw_id = str(job.get("id") or job.get("job_id") or "").strip()
    if raw_id:
        return f"indeed:{raw_id.replace('in-', '', 1)}"

    if raw_url:
        return f"indeed:{raw_url}"

    return f"indeed:{json.dumps(job, sort_keys=True, default=str)}"


def deduplicate_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard duplicate Indeed postings preserving the first occurrence."""
    seen: set[str] = set()
    unique_jobs: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        key = canonical_indeed_job_key(job)
        if key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)
    return unique_jobs


def _build_agent(api_key: str | None = None) -> Agent:
    google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError(
            "Provide api_key or set the GOOGLE_API_KEY environment variable."
        )

    provider = GoogleProvider(api_key=google_api_key)
    model = GoogleModel("gemini-flash-lite-latest", provider=provider)
    return Agent(
        model,
        output_type=Job,
        instructions=(
            "You are extracting structured job facts from a scraped Indeed posting. "
            "Use the raw scraped fields as authoritative when they are non-null. "
            "Never overwrite non-null scraped values with values from the description. "
            "When a scraped value is null or missing, use the description to fill it, but "
            "only when the information is explicit in the description. "
            "Do not infer facts that are not clearly stated. "
            "Do not invent salary endpoints, experience, licenses, skills, education, benefits, "
            "industry, shifts, travel, or remote status when the description does not explicitly say so. "
            "If the description says a salary is 'From $24.00 per hour', populate salary_min=24, "
            "salary_currency='USD', salary_interval='hour', and leave salary_max=None unless an explicit "
            "upper bound is stated. If a scraped salary is already present, preserve it exactly. "
            "For salary_interval use one of: 'hour', 'day', 'week', 'month', 'year'. "
            "Return only the structured Job output."
        ),
    )


class JobEnricher:
    """Extract missing job facts from the description using the existing Gemini setup."""

    def __init__(self, *, api_key: str | None = None, agent: Any | None = None, limiter: GeminiRateLimiter | None = None):
        self.agent = agent or _build_agent(api_key)
        self.limiter = limiter or get_shared_gemini_limiter()

    def enrich_job(self, job: dict[str, Any]) -> Job:
        raw_job = dict(job)
        # Preserve the original scraped record before the LLM may add missing details.
        prompt = (
            "Extract and enrich the job from the raw Indeed posting. "
            "Keep all non-null scraped values authoritative, and only fill missing fields from the description. "
            "Never infer facts that are not explicit.\n\n"
            f"Raw scraped job JSON:\n{json.dumps(raw_job, ensure_ascii=False, indent=2, default=str)}"
        )
        prompt = self.limiter.ensure_prompt_within_budget(prompt)
        self.limiter.acquire()
        result = self.agent.run_sync(prompt)
        enriched = result.output

        enriched.id = str(raw_job.get("id") or enriched.id or canonical_indeed_job_key(raw_job))
        enriched.source = str(raw_job.get("site") or raw_job.get("source") or enriched.source or "indeed")
        enriched.url = str(raw_job.get("job_url") or raw_job.get("url") or raw_job.get("job_url_direct") or enriched.url or "")
        enriched.title = str(raw_job.get("title") or enriched.title or "")
        enriched.company = str(raw_job.get("company") or enriched.company or "")
        enriched.location = raw_job.get("location") if raw_job.get("location") is not None else enriched.location
        enriched.date_posted = _parse_datetime(raw_job.get("date_posted")) or enriched.date_posted
        enriched.job_type = raw_job.get("job_type") if raw_job.get("job_type") is not None else enriched.job_type
        enriched.is_remote = raw_job.get("is_remote") if raw_job.get("is_remote") is not None else enriched.is_remote

        if raw_job.get("min_amount") is not None:
            enriched.salary_min = float(raw_job["min_amount"])
        if raw_job.get("max_amount") is not None:
            enriched.salary_max = float(raw_job["max_amount"])
        if raw_job.get("currency"):
            enriched.salary_currency = str(raw_job["currency"])
        if raw_job.get("interval"):
            enriched.salary_interval = _normalize_salary_interval(raw_job["interval"])

        enriched.description = str(raw_job.get("description") or enriched.description or "")
        enriched.search_term = str(raw_job.get("search_term") or enriched.search_term or "")
        enriched.scraped_at = datetime.now(timezone.utc)

        return enriched


def _normalize_salary_interval(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    mapping = {
        "hourly": "hour",
        "hour": "hour",
        "daily": "day",
        "day": "day",
        "weekly": "week",
        "week": "week",
        "monthly": "month",
        "month": "month",
        "yearly": "year",
        "year": "year",
    }
    return mapping.get(text, text)


def enrich_jobs_in_batches(
    jobs: Sequence[dict[str, Any]],
    *,
    batch_size: int = BATCH_SIZE,
    api_key: str | None = None,
) -> list[Job]:
    """Process jobs in batches of a configurable size and return enriched Job objects."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    enricher = JobEnricher(api_key=api_key)
    enriched_jobs: list[Job] = []
    for start in range(0, len(jobs), batch_size):
        batch = list(jobs[start : start + batch_size])
        for raw_job in batch:
            enriched_jobs.append(enricher.enrich_job(raw_job))
    return enriched_jobs


def store_jobs_in_sqlite(
    db_path: str | Path,
    raw_jobs: Sequence[dict[str, Any]],
    enriched_jobs: Sequence[dict[str, Any] | Job],
) -> Path:
    """Persist raw Indeed jobs and enriched Job records into SQLite."""
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_jobs (
                job_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                stored_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enriched_jobs (
                job_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                stored_at TEXT NOT NULL
            )
            """
        )

        now = datetime.now(timezone.utc).isoformat()
        for raw_job in raw_jobs:
            source = str(raw_job.get("site") or raw_job.get("source") or "indeed")
            job_key = canonical_indeed_job_key(raw_job)
            payload = json.dumps(raw_job, ensure_ascii=False, sort_keys=True, default=str)
            conn.execute(
                """
                INSERT INTO raw_jobs (job_key, source, payload_json, stored_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    source = excluded.source,
                    payload_json = excluded.payload_json,
                    stored_at = excluded.stored_at
                """,
                (job_key, source, payload, now),
            )

        for enriched_job in enriched_jobs:
            if isinstance(enriched_job, Job):
                payload = enriched_job.model_dump(mode="json")
                source = enriched_job.source
                job_key = canonical_indeed_job_key({
                    "site": source,
                    "job_url": enriched_job.url,
                    "id": enriched_job.id,
                })
            else:
                payload = dict(enriched_job)
                source = str(payload.get("source") or payload.get("site") or "indeed")
                job_key = canonical_indeed_job_key({
                    "site": source,
                    "job_url": payload.get("url") or payload.get("job_url"),
                    "id": payload.get("id"),
                })
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            conn.execute(
                """
                INSERT INTO enriched_jobs (job_key, source, payload_json, stored_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    source = excluded.source,
                    payload_json = excluded.payload_json,
                    stored_at = excluded.stored_at
                """,
                (job_key, source, payload_json, now),
            )

        conn.commit()
    finally:
        conn.close()

    gc.collect()
    return db


def read_jobs_json(path: str | Path) -> list[dict[str, Any]]:
    serialized = Path(path).read_text(encoding="utf-8")
    data = json.loads(serialized)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of jobs in {path}.")
    return data


def enrich_jobs_file(
    jobs_path: str | Path,
    *,
    db_path: str | Path,
    batch_size: int = BATCH_SIZE,
    api_key: str | None = None,
) -> list[Job]:
    raw_jobs = deduplicate_jobs(read_jobs_json(jobs_path))
    enriched_jobs = enrich_jobs_in_batches(raw_jobs, batch_size=batch_size, api_key=api_key)
    store_jobs_in_sqlite(db_path, raw_jobs, enriched_jobs)
    return enriched_jobs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich scraped Indeed jobs with a PydanticAI Gemini extraction agent and store them in SQLite."
    )
    parser.add_argument("jobs_path", type=Path, help="Path to the scraped jobs.json file.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("jobs.db"),
        help="Path for the SQLite database containing raw and enriched jobs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to process jobs in before writing results.",
    )
    parser.add_argument(
        "--api-key",
        help="Google API key; defaults to GOOGLE_API_KEY.",
    )
    args = parser.parse_args()
    enrich_jobs_file(args.jobs_path, db_path=args.db_path, batch_size=args.batch_size, api_key=args.api_key)


if __name__ == "__main__":
    main()
