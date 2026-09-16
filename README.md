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
