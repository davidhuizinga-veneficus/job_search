## CV-scoped job storage

The pipeline generates a new UUID for `cv_id` on every run. This means each
CV submission gets its own database grouping, even when the same job appears
for multiple CVs:

```text
uv run python -m job_search.pipeline cv.pdf --jobs-db-path jobs.db
```

Both `raw_jobs` and `enriched_jobs` contain `cv_id`. The database keeps the
same job separately for different CVs using `(cv_id, job_key)` as the key.
Retrieve one CV's records with:

```sql
SELECT * FROM enriched_jobs WHERE cv_id = 'generated-run-uuid';
```

The pipeline also prints a `scrape_id`. Rank that scrape with:

```text
uv run python -m job_search.rank_jobs --db-path jobs.db --cv-json-path cv.json --cv-id <cv_id> --scrape-id <scrape_id>
```

Ranking accepts structured selection filters such as `--from-date`, `--to-date`,
and `--remote-only`. Results are stored in `job_rankings`; retrieve them in
score order with:

```sql
SELECT * FROM job_rankings
WHERE ranking_run_id = '<ranking_run_id>'
ORDER BY score DESC;
```
