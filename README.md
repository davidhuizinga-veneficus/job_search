# JobAgent

JobAgent is a Streamlit app that finds jobs that fit your CV. You upload a PDF CV,
and the app:

1. reads the CV and turns it into structured data (skills, education, experience, location),
2. comes up with search terms that fit your profile,
3. searches Indeed for jobs near you,
4. extracts the key facts from each vacancy (salary, requirements, remote work, and so on),
5. scores every job against your CV and shows them ranked from best to worst match.

It uses Google's Gemini models for the AI steps, so you need a Gemini API key. The free tier is enough.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs the right Python version (3.13) and all dependencies for you.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- An internet connection. The app calls Gemini, searches Indeed, and looks up locations on OpenStreetMap.

## Setup

1. Clone the repository and open a terminal in the project folder.
2. Create a file called `.env` in the project folder with your API key:

   ```text
   GOOGLE_API_KEY=your-api-key-here
   ```

3. Install the dependencies:

   ```text
   uv sync
   ```

## Starting the app

Run this from the project folder:

```text
uv run streamlit run src/job_search/app.py
```

Streamlit opens the app in your browser, usually at <http://localhost:8501>.
Always start it from the project folder: that is where it finds `.env` and the
theme settings in `.streamlit/config.toml`.

## Using the app

1. **Upload your CV.** Click *Upload CV* and pick a PDF. The file name shows up
   below the upload box.
2. **Choose the settings.**
   - *Scrape radius (miles)*: how far from your home location to search. Default: 50.
   - *Results per search term*: how many vacancies to fetch for each search term.
     Default: 20. Every vacancy costs one Gemini request to analyse, so a lower
     number makes a run much faster. Use 1–3 for a quick test.
3. **Click Run.** A status box shows which step is running and a progress bar
   shows how far along it is. If the app has to wait for the Gemini rate
   limit, it shows a countdown.
4. **Read the results.** When the run finishes:
   - your location, as read from the CV, appears next to the file name;
   - a bar chart shows how the match scores are spread;
   - the jobs are listed as cards, best match first, 10 per page. Use
     *Previous* and *Next* to page through them.

### Reading a job card

Each card shows the job title, company, location, distance from your home,
salary (if the vacancy lists one) and the overall match score. The coloured dot
shows the recommendation:

| Dot | Meaning |
|---|---|
| Green | Strong match: you meet the key requirements |
| Orange | Possible match: some key requirements are missing, but it is worth a look |
| Red | Weak match |

Click **Apply →** to open the vacancy on Indeed in a new tab.

Click **View details** to see why the job got its score: a short summary plus a
score from 0 to 100 for each part of the match, each with an explanation.

| Part | What it checks |
|---|---|
| Skills | Required skills and tools compared with your CV |
| Education | Required degree or field of study |
| Experience | Years of experience in similar roles |
| Seniority | Whether your level fits the role (only relevant experience counts) |
| Licenses and certifications | Required licenses or certificates |
| Location | Distance to the job, and whether it is remote |
| Role match | How close the role is to what you have done before |
| Salary | Salary information in the vacancy |
| Benefits | Benefits mentioned in the vacancy |

The overall score is a weighted mix of these parts. Skills, experience and
education count most, salary and benefits least.

### Starting over

Upload a different CV to clear the results and start a new run. Refreshing the
browser page also clears the results. They stay saved in the database (see below).

## Where the data goes

Every run writes to files in the project folder:

| File | Contents |
|---|---|
| `cv.json` | The structured version of the last CV you uploaded |
| `search_terms.json` | The search terms generated for that CV |
| `jobs.json` | The raw vacancies found by the last search |
| `jobs.db` | A SQLite database with all vacancies and rankings from every run |

These files contain personal data from your CV and are listed in `.gitignore`.
Don't commit them.

## Rate limits and slow runs

The Gemini free tier allows a limited number of requests per minute. The app
stays under that limit by waiting when needed. By default it sends at most 15
requests per minute. If your model allows more (check your
[AI Studio rate-limit page](https://aistudio.google.com/rate-limit)), raise the
limit in `.env`:

```text
GEMINI_REQUESTS_PER_MINUTE=30
```

Most of a run's time is spent analysing vacancies, one request per job. As a
rough guide, a run takes at least *number of jobs ÷ requests per minute*
minutes.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Provide api_key or set the GOOGLE_API_KEY environment variable` | `.env` is missing or you started the app from another folder. Check the file and start the app from the project folder. |
| `503 UNAVAILABLE ... high demand` | Google's servers for the model are overloaded for everyone; it is not caused by your usage. Wait a while and try again. |
| `429 RESOURCE_EXHAUSTED` | You hit your own Gemini quota. Wait a minute (or until the next day for the daily limit), or lower `GEMINI_REQUESTS_PER_MINUTE`. |
| A run hangs on "Enriching job X of Y" | The Gemini client retries failed requests several times before giving up, which can take minutes per job when the model is overloaded. |
| Few or no results | Increase the scrape radius or the results per search term. |

## Command-line use

The same steps can run without the app, for example for scripting:

```text
uv run python -m job_search.pipeline cv.pdf --jobs-db-path jobs.db
uv run python -m job_search.rank_jobs --db-path jobs.db --cv-json-path cv.json --cv-id <cv_id> --scrape-id <scrape_id>
```

The pipeline prints the `cv_id` and `scrape_id` to pass to the ranking step.
Ranking also accepts `--from-date`, `--to-date` and `--remote-only`.
