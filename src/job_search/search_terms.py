"""Devise job-search terms from a structured CV."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from job_search.cv_to_json import CV

load_dotenv()


class SearchTerms(BaseModel):
    """Search queries suitable for a job board such as Indeed."""

    search_terms: list[str] = Field(min_length=2, max_length=5)


def devise_search_terms(
    cv: CV,
    output_path: str | Path,
    api_key: str | None = None,
) -> SearchTerms:
    """Read a CV JSON file and write AI-generated job-search terms."""
    
    google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError(
            "Provide api_key or set the GOOGLE_API_KEY environment variable."
        )

    provider = GoogleProvider(api_key=google_api_key)
    model = GoogleModel("gemini-flash-lite-latest", provider=provider)
    agent = Agent(
        model,
        output_type=SearchTerms,
        instructions=(
            "Devise between two and five concise, distinct Indeed search queries "
            "based only on the supplied CV. Combine the candidate's strongest "
            "job titles, skills, industries, and experience into realistic queries. "
            "Prefer terms that employers use in job titles. Do not include locations "
            "unless the CV contains one. Do not invent skills, qualifications, or "
            "seniority. Return only the structured result."
        ),
    )
    result = agent.run_sync(
        "Create Indeed search terms from this structured CV:\n\n"
        f"{cv.model_dump_json(indent=2)}"
    )
    search_terms = result.output
    Path(output_path).write_text(
        search_terms.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return search_terms