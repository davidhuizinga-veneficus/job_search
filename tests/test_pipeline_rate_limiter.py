import json
import shutil
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from job_search.cv_to_json import CV
from job_search.gemini_limiter import get_shared_gemini_limiter, reset_shared_gemini_limiter
from job_search.job_enrichment import Job
from job_search.pipeline import main
from job_search.search_terms import SearchTerms


class TestPipelineRateLimiter(unittest.TestCase):
    def test_shared_limiter_is_reused_across_pipeline_steps(self):
        limiter = reset_shared_gemini_limiter()
        self.assertIs(limiter, get_shared_gemini_limiter())

    def test_full_pipeline_uses_shared_limiter(self):
        limiter = reset_shared_gemini_limiter()
        cv = CV(
            professional_profile={"current_job_title": "Security Guard"},
            location={"city": "Los Angeles", "region": "CA", "country": "US"},
        )

        class FakeResult:
            def __init__(self, output):
                self.output = output

        class FakeSearchTerms(BaseModel):
            search_terms: list[str]

        def fake_run_sync(self, prompt):
            if "CV text" in prompt:
                return FakeResult(cv)
            if "Create Indeed search terms" in prompt:
                return FakeResult(FakeSearchTerms(search_terms=["security guard", "door attendant"]))
            if "Raw scraped job JSON" in prompt:
                return FakeResult(
                    Job(
                        id="in-123",
                        source="indeed",
                        url="https://www.indeed.com/viewjob?jk=123",
                        title="Security Guard",
                        company="Acme Security",
                        description="Pay: From $24.00 per hour",
                        search_term="security guard",
                        scraped_at=datetime.now(timezone.utc),
                        salary_min=24.0,
                        salary_currency="USD",
                        salary_interval="hour",
                    )
                )
            raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

        tmp_dir = tempfile.mkdtemp()
        try:
            jobs_dir = Path(tmp_dir)
            jobs_path = jobs_dir / "jobs.json"
            jobs_path.write_text(
                json.dumps([
                    {
                        "id": "in-123",
                        "site": "indeed",
                        "job_url": "https://www.indeed.com/viewjob?jk=123&utm_source=linkedin",
                        "title": "Security Guard",
                        "company": "Acme Security",
                        "description": "Pay: From $24.00 per hour",
                        "search_term": "security guard",
                    }
                ]),
                encoding="utf-8",
            )
            db_path = jobs_dir / "jobs.db"

            with patch("job_search.cv_to_json.extract_pdf_text", return_value="Security guard CV."), \
                 patch("job_search.cv_to_json.Agent.run_sync", new=fake_run_sync), \
                 patch("job_search.search_terms.Agent.run_sync", new=fake_run_sync), \
                 patch("job_search.job_enrichment.Agent.run_sync", new=fake_run_sync), \
                 patch("job_search.pipeline.search_indeed", return_value=None), \
                 patch.object(limiter, "acquire", wraps=limiter.acquire) as acquire_mock:
                with patch("sys.argv", [
                    "job-search",
                    "cv.pdf",
                    "--cv-json-path",
                    str(jobs_dir / "cv.json"),
                    "--search-terms-path",
                    str(jobs_dir / "search_terms.json"),
                    "--jobs-path",
                    str(jobs_path),
                    "--jobs-db-path",
                    str(db_path),
                    "--batch-size",
                    "10",
                ]):
                    with patch(
                        "job_search.pipeline.uuid.uuid4",
                        return_value=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    ):
                        main()

            with sqlite3.connect(db_path) as conn:
                raw_count = conn.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
                enriched_count = conn.execute("SELECT COUNT(*) FROM enriched_jobs").fetchone()[0]
                cv_ids = conn.execute("SELECT DISTINCT cv_id FROM raw_jobs").fetchall()

            self.assertGreaterEqual(acquire_mock.call_count, 3)
            self.assertEqual(raw_count, 1)
            self.assertEqual(enriched_count, 1)
            self.assertEqual(cv_ids, [("00000000-0000-0000-0000-000000000001",)])
            self.assertIs(get_shared_gemini_limiter(), limiter)
        finally:
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA wal_checkpoint(FULL)")
            except Exception:
                pass
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
