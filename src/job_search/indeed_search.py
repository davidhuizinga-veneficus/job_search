"""Search Indeed using JobSpy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jobspy import scrape_jobs

from job_search.search_terms import SearchTerms


def _normalize_indeed_country(country: str | None) -> str:
    value = (country or "").strip().lower()
    aliases = {
        "nl": "netherlands",
        "nederland": "netherlands",
        "us": "usa",
        "united states": "usa",
        "united states of america": "usa",
        "uk": "uk",
        "united kingdom": "uk",
    }
    if value in aliases:
        return aliases[value]
    if value in {
        "",
        "unknown",
        "not specified",
        "niet gespecificeerd",
        "onbekend",
        "n/a",
        "na",
    }:
        return "netherlands"
    return value


def search_indeed(
    search_terms: SearchTerms,
    output_path: str | Path,
    *,
    location: str = "",
    country_indeed: str = "Netherlands",
    results_wanted: int = 20,
    scrape_radius_miles: int = 50,
    scrape_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search Indeed for every term and write the combined results as JSON."""
    jobs: list[dict[str, Any]] = []
    normalized_country = _normalize_indeed_country(country_indeed)

    for search_term in search_terms.search_terms:
        results = scrape_jobs(
            site_name=["indeed"],
            search_term=search_term,
            location=location,
            country_indeed=normalized_country,
            distance=scrape_radius_miles,
            results_wanted=results_wanted,
            description_format="markdown",
            verbose=1,
        )
        for job in json.loads(results.to_json(orient="records", date_format="iso")):
            job["search_term"] = search_term
            if scrape_id is not None:
                job["scrape_id"] = scrape_id
            jobs.append(job)

    Path(output_path).write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return jobs