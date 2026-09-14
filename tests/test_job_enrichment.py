import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from job_search.job_enrichment import Job, JobEnricher, deduplicate_jobs, store_jobs_in_sqlite


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload

    def run_sync(self, _prompt):
        return type("Result", (), {"output": self.payload})()


class TestJobEnrichment(unittest.TestCase):
    def test_deduplicates_indeed_urls_with_tracking_params(self):
        jobs = [
            {
                "id": "in-123",
                "site": "indeed",
                "job_url": "https://www.indeed.com/viewjob?jk=123&utm_source=linkedin",
                "title": "Security Guard",
                "company": "Acme Security",
                "description": "Security guard work.",
                "search_term": "Security Guard",
            },
            {
                "id": "in-456",
                "site": "indeed",
                "job_url": "https://www.indeed.com/viewjob?jk=123&utm_campaign=ads",
                "title": "Security Guard",
                "company": "Acme Security",
                "description": "Security guard work.",
                "search_term": "Security Guard",
            },
        ]

        self.assertEqual(len(deduplicate_jobs(jobs)), 1)

    def test_store_jobs_in_sqlite_keeps_raw_and_enriched_records(self):
        raw_jobs = [
            {
                "id": "in-123",
                "site": "indeed",
                "job_url": "https://www.indeed.com/viewjob?jk=123&utm_source=linkedin",
                "title": "Security Guard",
                "company": "Acme Security",
                "description": "Pay: From $24.00 per hour",
                "search_term": "Security Guard",
            }
        ]

        enriched_jobs = [
            {
                "id": "in-123",
                "source": "indeed",
                "url": "https://www.indeed.com/viewjob?jk=123",
                "title": "Security Guard",
                "company": "Acme Security",
                "description": "Pay: From $24.00 per hour",
                "search_term": "Security Guard",
                "salary_min": 24.0,
                "salary_currency": "USD",
                "salary_interval": "hour",
                "summary": "Security guard role.",
            }
        ]

        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(tmp_dir) / "jobs.db"
            store_jobs_in_sqlite(db_path, raw_jobs, enriched_jobs)

            with sqlite3.connect(db_path) as conn:
                raw_rows = conn.execute(
                    "SELECT COUNT(*) FROM raw_jobs"
                ).fetchone()[0]
                enriched_rows = conn.execute(
                    "SELECT COUNT(*) FROM enriched_jobs"
                ).fetchone()[0]
                sample = conn.execute(
                    "SELECT payload_json FROM raw_jobs WHERE job_key = 'indeed:123'"
                ).fetchone()

            self.assertEqual(raw_rows, 1)
            self.assertEqual(enriched_rows, 1)
            self.assertIsNotNone(sample)
            payload = json.loads(sample[0])
            self.assertEqual(payload["id"], "in-123")
        finally:
            pass

    def test_enricher_keeps_structured_salary_when_present(self):
        job = {
            "id": "in-123",
            "site": "indeed",
            "job_url": "https://www.indeed.com/viewjob?jk=123",
            "title": "Security Guard",
            "company": "Acme Security",
            "location": "Pasadena, CA, US",
            "description": "Pay: $21.00 - $23.00 per hour",
            "min_amount": 21.0,
            "max_amount": 23.0,
            "currency": "USD",
            "interval": "hourly",
            "is_remote": False,
            "search_term": "Security Guard",
        }

        payload = Job(
            id="in-123",
            source="indeed",
            url="https://www.indeed.com/viewjob?jk=123",
            title="Security Guard",
            company="Acme Security",
            description="Pay: $21.00 - $23.00 per hour",
            search_term="Security Guard",
            scraped_at=datetime.now(timezone.utc),
            salary_min=21.0,
            salary_max=23.0,
            salary_currency="USD",
            salary_interval="hour",
        )

        enriched = JobEnricher(agent=FakeAgent(payload)).enrich_job(job)
        self.assertEqual(enriched.salary_min, 21.0)
        self.assertEqual(enriched.salary_max, 23.0)
        self.assertEqual(enriched.salary_currency, "USD")
        self.assertEqual(enriched.salary_interval, "hour")


if __name__ == "__main__":
    unittest.main()
