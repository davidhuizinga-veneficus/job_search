"""Run the job-search pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from job_search.cv_to_json import convert_cv_to_json
from job_search.indeed_search import search_indeed
from job_search.search_terms import devise_search_terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CV PDF and devise job-search terms from it."
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the CV PDF file.",
    )
    parser.add_argument(
        "--cv-json-path",
        type=Path,
        default=Path("cv.json"),
        help="Path where the structured CV JSON should be written.",
    )
    parser.add_argument(
        "--search-terms-path",
        type=Path,
        default=Path("search_terms.json"),
        help="Path where generated search terms should be written.",
    )
    parser.add_argument(
        "--jobs-path",
        type=Path,
        default=Path("jobs.json"),
        help="Path where Indeed jobs should be written.",
    )
    parser.add_argument(
        "--api-key",
        help="Google API key; defaults to GOOGLE_API_KEY.",
    )
    args = parser.parse_args()

    cv = convert_cv_to_json(args.pdf_path, args.cv_json_path, args.api_key)
    search_terms = devise_search_terms(cv, args.search_terms_path, args.api_key)
    location_parts = [cv.location.city, cv.location.region]
    location = ", ".join(part for part in location_parts if part)
    country = cv.location.country or "Netherlands"
    search_indeed(
        search_terms,
        args.jobs_path,
        location=location,
        country_indeed=country,
    )


if __name__ == "__main__":
    main()