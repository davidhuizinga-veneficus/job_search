"""Rank enriched jobs against a structured CV."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from job_search.cv_to_json import CV
from job_search.gemini_limiter import get_shared_gemini_limiter

DEFAULT_WEIGHTS = {
    "skills": 0.22,
    "education": 0.14,
    "experience": 0.16,
    "seniority": 0.10,
    "licenses": 0.10,
    "location": 0.12,
    "role_match": 0.10,
    "salary": 0.04,
    "benefits": 0.02,
}


class RankingResult(BaseModel):
    job_key: str
    score: float = Field(ge=0, le=100)
    skills_score: float = Field(ge=0, le=100)
    education_score: float = Field(ge=0, le=100)
    experience_score: float = Field(ge=0, le=100)
    seniority_score: float = Field(ge=0, le=100)
    licenses_score: float = Field(ge=0, le=100)
    location_score: float = Field(ge=0, le=100)
    role_match_score: float = Field(ge=0, le=100)
    salary_score: float = Field(ge=0, le=100)
    benefits_score: float = Field(ge=0, le=100)
    recommendation: str
    recommendation_explanation: str
    matches: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    component_explanations: dict[str, str] = Field(default_factory=dict)


class RankingBatch(BaseModel):
    results: list[RankingResult]


@dataclass(frozen=True)
class RankingFilters:
    cv_id: str
    scrape_id: str
    from_date: str | None = None
    to_date: str | None = None
    remote_only: bool = False


def _geocode(conn: sqlite3.Connection, geocoder: Nominatim, location: str) -> tuple[float, float] | None:
    normalized = location.strip().lower()
    if not normalized:
        return None
    cached = conn.execute(
        "SELECT latitude, longitude FROM geocode_cache WHERE location = ?", (normalized,)
    ).fetchone()
    if cached:
        return float(cached[0]), float(cached[1])
    try:
        result = geocoder.geocode(location, exactly_one=True, timeout=10)
    except Exception:
        result = None
    if result is None:
        return None
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(location, latitude, longitude, stored_at) VALUES (?, ?, ?, ?)",
        (normalized, result.latitude, result.longitude, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return result.latitude, result.longitude


def _miles_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    radius = 3958.8
    first_lat, first_lon = map(math.radians, first)
    second_lat, second_lon = map(math.radians, second)
    delta_lat = second_lat - first_lat
    delta_lon = second_lon - first_lon
    value = math.sin(delta_lat / 2) ** 2 + math.cos(first_lat) * math.cos(second_lat) * math.sin(delta_lon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(value))


def _select_jobs(conn: sqlite3.Connection, filters: RankingFilters) -> list[dict[str, Any]]:
    clauses = ["cv_id = ?", "scrape_id = ?"]
    values: list[Any] = [filters.cv_id, filters.scrape_id]
    if filters.from_date:
        clauses.append("date_posted >= ?")
        values.append(filters.from_date)
    if filters.to_date:
        clauses.append("date_posted <= ?")
        values.append(filters.to_date)
    if filters.remote_only:
        clauses.append("is_remote = 1")
    query = "SELECT * FROM enriched_jobs WHERE " + " AND ".join(clauses)
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(query, values)]


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS geocode_cache (location TEXT PRIMARY KEY, latitude REAL NOT NULL, longitude REAL NOT NULL, stored_at TEXT NOT NULL)")
    conn.execute("""CREATE TABLE IF NOT EXISTS job_rankings (
        ranking_run_id TEXT NOT NULL, cv_id TEXT NOT NULL, scrape_id TEXT NOT NULL, job_key TEXT NOT NULL,
        vacancy_url TEXT NOT NULL,
        score REAL NOT NULL, skills_score REAL NOT NULL, education_score REAL NOT NULL,
        experience_score REAL NOT NULL, seniority_score REAL NOT NULL, licenses_score REAL NOT NULL,
        location_score REAL NOT NULL, role_match_score REAL NOT NULL, salary_score REAL NOT NULL,
        benefits_score REAL NOT NULL, distance_miles REAL, location_eligible BOOLEAN,
        recommendation TEXT NOT NULL, recommendation_explanation TEXT NOT NULL,
        matches TEXT NOT NULL, missing_requirements TEXT NOT NULL, conflicts TEXT NOT NULL,
        component_explanations TEXT NOT NULL, model TEXT NOT NULL, criteria_version TEXT NOT NULL,
        weights TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (ranking_run_id, cv_id, scrape_id, job_key)
    )""")
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_rankings)")
    }
    if "vacancy_url" not in existing_columns:
        conn.execute("ALTER TABLE job_rankings ADD COLUMN vacancy_url TEXT")


def _rank_batch(
    agent: Agent,
    limiter: Any,
    cv: CV,
    batch: list[dict[str, Any]],
) -> list[RankingResult]:
    """Rank a batch, splitting it when the model returns an incomplete batch."""
    prompt = json.dumps(
        {
            "cv": cv.model_dump(mode="json"),
            "jobs": [
                {"batch_position": position, **job}
                for position, job in enumerate(batch)
            ],
            "weights": DEFAULT_WEIGHTS,
        },
        ensure_ascii=False,
        default=str,
    )
    prompt = limiter.ensure_prompt_within_budget(prompt)
    limiter.acquire()
    result = agent.run_sync(prompt).output
    expected_keys = [job["job_key"] for job in batch]
    received_keys = [item.job_key for item in result.results]
    if len(received_keys) == len(expected_keys):
        if received_keys != expected_keys:
            for item, expected_key in zip(result.results, expected_keys):
                item.job_key = expected_key
        return result.results

    if len(batch) > 1:
        midpoint = len(batch) // 2
        return _rank_batch(agent, limiter, cv, batch[:midpoint]) + _rank_batch(
            agent, limiter, cv, batch[midpoint:]
        )

    raise ValueError(
        "Ranking model returned the wrong number of results for one job: "
        f"expected 1, received {len(received_keys)}."
    )


def rank_jobs(
    db_path: str | Path,
    cv_json_path: str | Path,
    filters: RankingFilters,
    *,
    max_distance_miles: float = 50,
    batch_size: int = 5,
    api_key: str | None = None,
) -> str:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if max_distance_miles < 0:
        raise ValueError("max_distance_miles must not be negative.")
    cv = CV.model_validate_json(Path(cv_json_path).read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(db_path))
    try:
        _create_tables(conn)
        jobs = _select_jobs(conn, filters)
        cv_location = ", ".join(part for part in (cv.location.city, cv.location.region, cv.location.country) if part)
        geocoder = Nominatim(user_agent="job-search-poc")
        cv_coordinates = _geocode(conn, geocoder, cv_location)
        for job in jobs:
            coordinates = _geocode(conn, geocoder, str(job.get("location") or ""))
            distance = _miles_between(cv_coordinates, coordinates) if cv_coordinates and coordinates else None
            job["distance_miles"] = round(distance, 1) if distance is not None else None
            job["location_eligible"] = bool(
                distance is not None and distance <= max_distance_miles
            ) or bool(job.get("is_remote") is True and distance is not None and distance > max_distance_miles)
            if distance is None:
                job["location_eligible"] = None

        provider = GoogleProvider(api_key=api_key)
        agent = Agent(
            GoogleModel("gemini-flash-lite-latest", provider=provider),
            output_type=RankingBatch,
            instructions="Rank each job against the complete structured CV. Return exactly one result per job_key. Treat the CV as complete: missing required evidence is a failure. Use the provided location score and eligibility facts; do not estimate distance. Explain every component concisely with readable keyword-style evidence. Scores must be 0-100.",
        )
        limiter = get_shared_gemini_limiter()
        ranking_run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            for item, job in zip(_rank_batch(agent, limiter, cv, batch), batch):
                conn.execute(
                    "INSERT INTO job_rankings (ranking_run_id, cv_id, scrape_id, job_key, vacancy_url, score, skills_score, education_score, experience_score, seniority_score, licenses_score, location_score, role_match_score, salary_score, benefits_score, distance_miles, location_eligible, recommendation, recommendation_explanation, matches, missing_requirements, conflicts, component_explanations, model, criteria_version, weights, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ranking_run_id, filters.cv_id, filters.scrape_id, item.job_key, job.get("url"), item.score, item.skills_score, item.education_score, item.experience_score, item.seniority_score, item.licenses_score, item.location_score, item.role_match_score, item.salary_score, item.benefits_score, job["distance_miles"], job["location_eligible"], item.recommendation, item.recommendation_explanation, json.dumps(item.matches), json.dumps(item.missing_requirements), json.dumps(item.conflicts), json.dumps(item.component_explanations), "gemini-flash-lite-latest", "v1", json.dumps(DEFAULT_WEIGHTS), created_at),
                )
        conn.commit()
        return ranking_run_id
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank enriched jobs against a CV.")
    parser.add_argument("--db-path", type=Path, default=Path("jobs.db"))
    parser.add_argument("--cv-json-path", type=Path, default=Path("cv.json"))
    parser.add_argument("--cv-id", required=True)
    parser.add_argument("--scrape-id", required=True)
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--max-distance-miles", type=float, default=50)
    parser.add_argument("--remote-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    run_id = rank_jobs(args.db_path, args.cv_json_path, RankingFilters(args.cv_id, args.scrape_id, args.from_date, args.to_date, args.remote_only), max_distance_miles=args.max_distance_miles, batch_size=args.batch_size, api_key=args.api_key)
    print(f"ranking_run_id={run_id}")


if __name__ == "__main__":
    main()
