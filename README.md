# Candidate Data Transformation Engine

Extracts, normalises, merges, and projects candidate profiles from multiple input sources
into a single canonical output with confidence scores and full provenance.

See [DESIGN.md](./DESIGN.md) for the full technical design document.

---

## Quick start

### Step 0: Clone the Repository

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/sharma-anahita/candidate-transformer.git
cd candidate-transformer
```

---

### Step 1: Check Python Version

**You need Python 3.11 or higher.** Check your version:

```bash
python --version
```

### Option 1: Running with a Virtual Environment (Recommended)

Using a virtual environment keeps project dependencies isolated:

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies & run app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies & run app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

### Option 2: Running with Global Python

If you prefer to install packages directly to your global environment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

> On some systems `python` or `pip` points to Python 2. If so, replace them with `python3` and `pip3` respectively.

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

## LLM Resume Parser (optional)

The resume parser supports advanced, high-accuracy extraction using Large Language Models (LLMs) via the OpenAI SDK wrapper. 

* **LLM Extraction**: If a `GROQ_API_KEY` is provided, the parser automatically uses the `llama-3.3-70b-versatile` model to extract structured candidate details (skills, experience, contact info, education, etc.) from PDF resumes.
* **Heuristic Fallback**: If no LLM key is configured, the parser automatically falls back to the built-in rule-based/heuristic parsing mechanism (regex, line analysis, and keyword matching) to extract details without requiring any external APIs.

### Setup

Add your key to the `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

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


