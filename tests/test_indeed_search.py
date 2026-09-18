import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from job_search.indeed_search import search_indeed
from job_search.search_terms import SearchTerms


class FakeResults:
    def to_json(self, **_kwargs):
        return "[]"


class TestIndeedSearch(unittest.TestCase):
    def test_placeholder_country_falls_back_to_netherlands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "jobs.json"
            with patch(
                "job_search.indeed_search.scrape_jobs",
                return_value=FakeResults(),
            ) as scrape:
                search_indeed(
                    SearchTerms(search_terms=["software engineer", "backend engineer"]),
                    output_path,
                    country_indeed="niet gespecificeerd",
                )

        self.assertEqual(scrape.call_args.kwargs["country_indeed"], "netherlands")


if __name__ == "__main__":
    unittest.main()
