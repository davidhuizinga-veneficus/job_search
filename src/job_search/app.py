"""Streamlit UI for running the current CV scrape and viewing its ranking."""

from __future__ import annotations

import math
import time
import uuid
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import streamlit as st

from job_search.cv_to_json import calculate_cv_id, convert_cv_to_json
from job_search.gemini_limiter import get_shared_gemini_limiter
from job_search.indeed_search import search_indeed
from job_search.job_enrichment import (
    deduplicate_jobs,
    enrich_jobs_in_batches,
    read_jobs_json,
    store_jobs_in_sqlite,
)
from job_search.rank_jobs import RankingFilters, rank_jobs
from job_search.search_terms import devise_search_terms

DB_PATH = Path("jobs.db")
CV_JSON_PATH = Path("cv.json")
JOBS_PATH = Path("jobs.json")
RESULTS_PER_PAGE = 10


def _query_rankings(db_path: Path, ranking_run_id: str) -> list[dict[str, Any]]:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM job_rankings WHERE ranking_run_id = ? ORDER BY score DESC",
                (ranking_run_id,),
            )
        ]
    finally:
        conn.close()


def _wait_callback(waiting: Any) -> Any:
    def update(wait_seconds: float) -> None:
        remaining = max(0, math.ceil(wait_seconds))
        while remaining:
            waiting.info(f"Waiting for model rate limit: {remaining} seconds")
            time.sleep(1)
            remaining -= 1
        waiting.empty()

    return update


def _recommendation_color(recommendation: str) -> str:
    text = recommendation.lower()
    if "strong" in text:
        return "#198754"
    if any(word in text for word in ("review", "possible", "moderate")):
        return "#d98c00"
    if any(word in text for word in ("weak", "poor", "not")):
        return "#c0392b"
    return "#6c757d"


def _format_salary(row: dict[str, Any]) -> str:
    minimum = row.get("salary_min")
    maximum = row.get("salary_max")
    currency = row.get("salary_currency") or ""
    interval = row.get("salary_interval") or ""
    if minimum is None and maximum is None:
        return "Salary unavailable"
    if minimum is not None and maximum is not None:
        value = f"{minimum:g}-{maximum:g}"
    else:
        value = f"{minimum if minimum is not None else maximum:g}"
    return f"{currency} {value} / {interval}".strip(" /")


def _render_card(row: dict[str, Any], rank: int) -> None:
    recommendation = str(row.get("recommendation") or "Unknown")
    color = _recommendation_color(recommendation)
    distance = row.get("distance_miles")
    distance_text = f"{distance:g} miles" if distance is not None else "Distance unavailable"
    st.markdown(
        f"""<div class="job-card">
        <div class="job-card-heading"><span class="rank">#{rank}</span><span class="marker" style="background:{color}"></span><strong>{row.get('title') or 'Untitled role'}</strong><span class="score">{row.get('score', 0):g}%</span></div>
        <div class="muted">{row.get('company') or 'Company unavailable'} · {row.get('location') or 'Location unavailable'} · {distance_text}</div>
        <div class="muted">{_format_salary(row)}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("View details", expanded=False):
        st.markdown(f"**{recommendation}**")
        st.write(row.get("recommendation_explanation") or "No explanation available.")
        component_names = (
            "skills", "education", "experience", "seniority", "licenses",
            "location", "role_match", "salary", "benefits",
        )
        explanations = _json_object(row.get("component_explanations"))
        for component in component_names:
            score = row.get(f"{component}_score")
            if score is not None:
                st.markdown(f"**{component.replace('_', ' ').title()}**: {score:g} / 100")
                if explanations.get(component):
                    st.caption(explanations[component])
        if row.get("vacancy_url"):
            st.link_button("Apply →", row["vacancy_url"], use_container_width=False)
        else:
            st.button("Vacancy link unavailable", disabled=True, key=f"missing-{row['job_key']}")


def _json_object(value: Any) -> dict[str, str]:
    import json

    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _run_pipeline(
    uploaded_file: Any,
    scrape_radius: int,
    results_wanted: int,
    status: Any,
    progress: Any,
    waiting: Any,
) -> None:
    with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        pdf_path = Path(handle.name)
    cv_id = calculate_cv_id(pdf_path)
    scrape_id = str(uuid.uuid4())
    limiter = get_shared_gemini_limiter()
    previous_callback = limiter.wait_callback
    limiter.wait_callback = _wait_callback(waiting)
    try:
        status.update(label="Preparing CV", state="running")
        cv = convert_cv_to_json(pdf_path, CV_JSON_PATH)
        progress.progress(0.15)
        status.update(label="Generating search terms", state="running")
        search_terms = devise_search_terms(cv, Path("search_terms.json"))
        progress.progress(0.25)
        status.update(label="Searching for jobs", state="running")
        search_indeed(
            search_terms,
            JOBS_PATH,
            location=", ".join(part for part in (cv.location.city, cv.location.region) if part),
            country_indeed=cv.location.country or "Netherlands",
            results_wanted=results_wanted,
            scrape_radius_miles=scrape_radius,
            scrape_id=scrape_id,
        )
        progress.progress(0.35)
        raw_jobs = deduplicate_jobs(read_jobs_json(JOBS_PATH))
        status.update(label="Enriching jobs", state="running")
        enriched_jobs = enrich_jobs_in_batches(
            raw_jobs,
            api_key=None,
            progress_callback=lambda completed, total: (
                status.write(f"Enriching job {completed} of {total}"),
                progress.progress(0.35 + 0.35 * completed / max(total, 1)),
            ),
        )
        store_jobs_in_sqlite(DB_PATH, raw_jobs, enriched_jobs, cv_id=cv_id, scrape_id=scrape_id)
        status.update(label="Ranking jobs", state="running")
        ranking_run_id = rank_jobs(
            DB_PATH,
            CV_JSON_PATH,
            RankingFilters(cv_id, scrape_id),
            max_distance_miles=50,
            api_key=None,
            progress_callback=lambda completed, total: (
                status.write(f"Ranking batch {completed} of {total}"),
                progress.progress(0.70 + 0.30 * completed / max(total, 1)),
            ),
        )
        progress.progress(1.0)
        st.session_state.update(
            ranking_run_id=ranking_run_id,
            profile_name=uploaded_file.name,
            profile_location=", ".join(part for part in (cv.location.city, cv.location.region) if part),
            results=_query_rankings(DB_PATH, ranking_run_id),
        )
        status.update(label="Finished", state="complete")
    finally:
        limiter.wait_callback = previous_callback
        pdf_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="JobAgent", page_icon="🤖", layout="wide")
    st.markdown("""<style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    .job-card { border: 1px solid #dfe4e8; border-radius: 8px; padding: 1rem 1.2rem .8rem; margin-top: .8rem; background: #fff; }
    .job-card-heading { display: flex; align-items: center; gap: .65rem; font-size: 1.05rem; }
    .rank, .score { color: #53616b; font-variant-numeric: tabular-nums; }
    .score { margin-left: auto; font-weight: 700; color: #17212b; }
    .marker { width: .75rem; height: .75rem; border-radius: 50%; display: inline-block; }
    .muted { color: #53616b; margin-top: .35rem; }
    </style>""", unsafe_allow_html=True)
    st.title("🤖 JobAgent")

    uploaded_file = st.file_uploader("📄 Upload CV", type=["pdf"])
    if not uploaded_file:
        st.session_state.pop("results", None)
        st.info("Upload a PDF CV to begin.")
        return

    upload_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("upload_id") != upload_id:
        st.session_state.pop("results", None)
        st.session_state.pop("page", None)
        st.session_state.upload_id = upload_id
    left, right = st.columns(2)
    with left:
        st.markdown(f"**📄 {uploaded_file.name}**")
    with right:
        st.markdown("**📍 Location will be shown after running**")
    scrape_radius = st.number_input("Scrape radius (miles)", min_value=1, value=50)
    results_wanted = st.number_input("Results per search term", min_value=1, value=20)
    run = st.button("Run", type="primary", disabled=st.session_state.get("running", False), use_container_width=True)
    if run:
        st.session_state.running = True
        try:
            with st.status("Starting", expanded=True) as status:
                progress = st.progress(0)
                waiting = st.empty()
                _run_pipeline(
                    uploaded_file,
                    int(scrape_radius),
                    int(results_wanted),
                    status,
                    progress,
                    waiting,
                )
        except Exception as error:
            st.error(f"The run failed: {error}")
        finally:
            st.session_state.running = False

    results = st.session_state.get("results")
    if not results:
        return

    st.markdown(f"**📄 {st.session_state['profile_name']}** · **📍 {st.session_state['profile_location'] or 'Unknown'}**")
    scores = [float(row["score"]) for row in results]
    bins = list(range(0, 101, 10))
    counts = [sum(1 for score in scores if lower <= score < lower + 10) for lower in bins[:-1]]
    histogram = pd.DataFrame({"Score range": [f"{lower}-{lower + 9}" for lower in bins[:-1]], "Jobs": counts})
    st.bar_chart(histogram, x="Score range", y="Jobs", color="#5b7c99")
    st.caption("Overall scrape summary will appear here.")
    minimum_score = st.slider("Minimum match", 0, 100, 75)
    filtered = [row for row in results if float(row["score"]) >= minimum_score]
    page_count = max(1, math.ceil(len(filtered) / RESULTS_PER_PAGE))
    page = min(st.session_state.get("page", 0), page_count - 1)
    previous, page_label, next_page = st.columns([1, 2, 1])
    with previous:
        if st.button("Previous", disabled=page == 0):
            st.session_state.page = page - 1
            st.rerun()
    with page_label:
        st.write(f"Page {page + 1} of {page_count}")
    with next_page:
        if st.button("Next", disabled=page >= page_count - 1):
            st.session_state.page = page + 1
            st.rerun()
    start = page * RESULTS_PER_PAGE
    for rank, row in enumerate(filtered[start : start + RESULTS_PER_PAGE], start=start + 1):
        _render_card(row, rank)


if __name__ == "__main__":
    main()
