# Candidate Data Transformation Engine

Extracts, normalises, merges, and projects candidate profiles from multiple input sources
into a single canonical output with confidence scores and full provenance.

See [DESIGN.md](./DESIGN.md) for the full technical design document.

---

## Quick start

**You need Python 3.11 or higher.** Check your version:

```bash
python --version
```

If it shows 3.11 or above, run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

That's it. The app opens at **http://localhost:8501** in your browser automatically.

> On some systems `python` points to Python 2. If so, use `python3` and `pip3` instead:
> ```bash
> python3 --version
> pip3 install -r requirements.txt
> streamlit run streamlit_app.py
> ```

---

## One-click scripts

**Windows** — double-click `run.bat` or run in terminal:

```
run.bat
```

**macOS / Linux** — run in terminal:

```bash
bash run.sh
```

These scripts install dependencies and start the app in one step.

---

## Trying it out immediately

Sample input files are in the `samples/` folder. You can upload them directly in the UI
without needing any live data or API keys:

| File | Upload as |
|---|---|
| `samples/sample_candidates.csv` | **Upload Candidates (CSV)** |
| `samples/greenhouse_candidate.json` | **Upload ATS Candidate (JSON)** |
| `samples/lever_candidate.json` | **Upload ATS Candidate (JSON)** |

1. Open the app at `http://localhost:8501`
2. Upload one or more of the sample files above
3. Click **Generate Output**
4. The projected candidate JSON appears at the bottom of the page

No API key is needed to run the CSV or ATS JSON sources.

---

## GitHub API key (optional)

The GitHub source requires a GitHub Personal Access Token to work beyond the free-tier
limit of 60 requests/hour (enough for ~3 candidates).

**Getting a token:**
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Select scopes: `read:user` and `public_repo`
4. Copy the token

**Using the token:**

You can paste it directly in the app under **API Secrets → GitHub Personal Access Token**
on the right side of the UI — no file setup needed.

Alternatively, create a `.env` file in the project root:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```env
GITHUB_TOKEN=your_token_here
```

> `.env` is listed in `.gitignore` and will never be committed.

---

## Running the tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Project structure

```
candidate-transformer/
├── src/
│   ├── adapters/          # One adapter per source type
│   ├── engines/           # MergeEngine, ConflictResolver, ConfidenceEngine
│   ├── models/            # Pydantic models (ExtractedCandidate, CanonicalCandidate, …)
│   ├── normalizers/       # PhoneNormalizer, SkillNormalizer, DateNormalizer, …
│   ├── projection/        # ProjectionEngine + JSONSchemaValidator
│   └── provenance/        # ProvenanceTracker
├── tests/                 # pytest test suite
├── samples/               # Sample input files (CSV, JSON, resume)
├── streamlit_app.py       # Streamlit UI
├── DESIGN.md              # Technical design document
├── requirements.txt
└── pyproject.toml
```


