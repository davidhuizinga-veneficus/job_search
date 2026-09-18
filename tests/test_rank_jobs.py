import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from job_search.job_enrichment import store_jobs_in_sqlite
from job_search.rank_jobs import RankingBatch, RankingFilters, RankingResult, rank_jobs


class FakeRankingAgent:
    def __init__(self, _model, output_type, instructions):
        self.output_type = output_type

    def run_sync(self, prompt):
        jobs = json.loads(prompt)["jobs"]
        return type(
            "Result",
            (),
            {
                "output": RankingBatch(
                    results=[
                        RankingResult(
                            job_key=job["job_key"],
                            score=80,
                            skills_score=90,
                            education_score=80,
                            experience_score=80,
                            seniority_score=70,
                            licenses_score=100,
                            location_score=90,
                            role_match_score=80,
                            salary_score=60,
                            benefits_score=50,
                            recommendation="strong_match",
                            recommendation_explanation="Strong skills and experience match.",
                            matches=["Python", "Backend development"],
                            missing_requirements=["AWS"],
                            component_explanations={"skills": "Python is explicitly listed."},
                        )
                        for job in jobs
                    ]
                )
            },
        )()


class MislabelingRankingAgent(FakeRankingAgent):
    def run_sync(self, prompt):
        result = super().run_sync(prompt)
        for item in result.output.results:
            item.job_key = "model-generated-id"
        return result


class IncompleteRankingAgent(FakeRankingAgent):
    def run_sync(self, prompt):
        result = super().run_sync(prompt)
        if len(result.output.results) > 1:
            result.output.results = result.output.results[:1]
        return result


class TestRankJobs(unittest.TestCase):
    def test_ranks_only_selected_scrape_and_persists_structured_result(self):
        raw_job = {
            "id": "in-123",
            "site": "indeed",
            "job_url": "https://www.indeed.com/viewjob?jk=123",
            "title": "Backend Developer",
            "company": "Acme",
            "location": "Amsterdam",
            "description": "Python backend role.",
            "search_term": "backend developer",
        }
        enriched_job = {
            "id": "in-123",
            "source": "indeed",
            "url": raw_job["job_url"],
            "title": raw_job["title"],
            "company": raw_job["company"],
            "location": raw_job["location"],
            "description": raw_job["description"],
            "search_term": raw_job["search_term"],
        }
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(tmp_dir) / "jobs.db"
            cv_path = Path(tmp_dir) / "cv.json"
            cv_path.write_text(json.dumps({"location": {"city": "Amsterdam"}}), encoding="utf-8")
            store_jobs_in_sqlite(db_path, [raw_job], [enriched_job], cv_id="cv-1", scrape_id="scrape-1")
            store_jobs_in_sqlite(db_path, [raw_job], [enriched_job], cv_id="cv-1", scrape_id="scrape-2")

            with patch("job_search.rank_jobs.Agent", FakeRankingAgent), patch(
                "job_search.rank_jobs._geocode", return_value=(52.37, 4.90)
            ):
                run_id = rank_jobs(
                    db_path,
                    cv_path,
                    RankingFilters("cv-1", "scrape-1"),
                    api_key="test-key",
                )

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT cv_id, scrape_id, vacancy_url, score, recommendation, matches FROM job_rankings"
                ).fetchall()

            self.assertTrue(run_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][:5], ("cv-1", "scrape-1", raw_job["job_url"], 80.0, "strong_match"))
            self.assertIn("Python", rows[0][5])
        finally:
            pass

    def test_ranking_aligns_complete_batch_when_model_mangles_job_keys(self):
        raw_job = {
            "id": "in-123",
            "site": "indeed",
            "job_url": "https://www.indeed.com/viewjob?jk=123",
            "title": "Backend Developer",
            "company": "Acme",
            "location": "Amsterdam",
            "description": "Python backend role.",
            "search_term": "backend developer",
        }
        enriched_job = {
            "id": "in-123",
            "source": "indeed",
            "url": raw_job["job_url"],
            "title": raw_job["title"],
            "company": raw_job["company"],
            "description": raw_job["description"],
            "search_term": raw_job["search_term"],
        }
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(tmp_dir) / "jobs.db"
            cv_path = Path(tmp_dir) / "cv.json"
            cv_path.write_text(json.dumps({"location": {"city": "Amsterdam"}}), encoding="utf-8")
            store_jobs_in_sqlite(db_path, [raw_job], [enriched_job], cv_id="cv-1", scrape_id="scrape-1")
            with patch("job_search.rank_jobs.Agent", MislabelingRankingAgent), patch(
                "job_search.rank_jobs._geocode", return_value=(52.37, 4.90)
            ):
                rank_jobs(db_path, cv_path, RankingFilters("cv-1", "scrape-1"), api_key="test-key")
            with sqlite3.connect(db_path) as conn:
                self.assertEqual(
                    conn.execute("SELECT job_key FROM job_rankings").fetchone()[0],
                    "indeed:123",
                )
        finally:
            pass

    def test_ranking_splits_incomplete_batches(self):
        raw_jobs = []
        enriched_jobs = []
        for job_number in range(2):
            raw_job = {
                "id": f"in-{job_number}",
                "site": "indeed",
                "job_url": f"https://www.indeed.com/viewjob?jk={job_number}",
                "title": "Backend Developer",
                "company": "Acme",
                "location": "Amsterdam",
                "description": "Python backend role.",
                "search_term": "backend developer",
            }
            raw_jobs.append(raw_job)
            enriched_jobs.append({
                "id": raw_job["id"], "source": "indeed", "url": raw_job["job_url"],
                "title": raw_job["title"], "company": raw_job["company"],
                "description": raw_job["description"], "search_term": raw_job["search_term"],
            })
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(tmp_dir) / "jobs.db"
            cv_path = Path(tmp_dir) / "cv.json"
            cv_path.write_text(json.dumps({"location": {"city": "Amsterdam"}}), encoding="utf-8")
            store_jobs_in_sqlite(db_path, raw_jobs, enriched_jobs, cv_id="cv-1", scrape_id="scrape-1")
            with patch("job_search.rank_jobs.Agent", IncompleteRankingAgent), patch(
                "job_search.rank_jobs._geocode", return_value=(52.37, 4.90)
            ):
                rank_jobs(db_path, cv_path, RankingFilters("cv-1", "scrape-1"), api_key="test-key")
            with sqlite3.connect(db_path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM job_rankings").fetchone()[0], 2)
        finally:
            pass


if __name__ == "__main__":
    unittest.main()
