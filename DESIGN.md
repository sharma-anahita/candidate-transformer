# Technical Design — Multi-Source Candidate Data Transformer

**Author:** Candidate submission · **Version:** 1.0.0

---

## Pipeline

```
Source Input
    │
    ├─ CSVAdapter          ┐
    ├─ ATSJsonAdapter      │  Extract → ExtractedCandidate
    ├─ ResumeAdapter       │  (raw, unnormalised; partial data is valid output)
    ├─ GitHubAdapter       │
    ├─ LinkedInAdapter     │
    └─ RecruiterNotesAdapter ┘
           │
    CandidateNormalizer          Normalize → NormalizedCandidate
    (phone, email, name, date, skill, company, education, location)
           │
    MergeEngine                  Merge → CanonicalCandidate (confidence=0.0 placeholders)
    (deduplicate + collect conflicts)
           │
    ConflictResolver             Resolve scalar field conflicts
    (5 deterministic rules)
           │
    ConfidenceEngine             Score → confidence per field + overall_confidence
    (source_weight × method_weight × quality_multiplier + bonuses)
           │
    ProjectionEngine             Project → custom JSON shape (runtime config)
           │
    JSONSchemaValidator          Validate → schema-valid output or structured error
```

Each adapter returns exactly one `ExtractedCandidate`. Adapters never raise on missing
field data — they call `add_warning()` and continue. `AdapterError` is reserved for
unrecoverable failures (auth failure, unreadable file). One bad source does not block others.

---

## Canonical Schema

The central output type is `CanonicalCandidate`. Key design decisions:

**Scalar fields** (`first_name`, `last_name`, `summary`, `location`) are typed as
`Optional[ConfidenceField[T]]` — a generic Pydantic model that co-locates the value,
its confidence score, its full provenance chain, and any losing conflict values. This
keeps confidence and lineage structurally bound to the value they describe.

**List fields** (`emails`, `phones`, `skills`, `experience`, `education`) use
`Canonical*` submodels (e.g., `CanonicalSkill(Skill)`) that add `confidence: float`
and `provenance: list[Provenance]` per item. Deduplication happens at merge time.

**Provenance** is immutable (`model_config = ConfigDict(frozen=True)`). Every `Provenance`
record carries: `source_type`, `adapter_name`, `method` (ExtractionMethod enum),
`source_id`, `raw_value` (the un-normalised original), `extracted_at` (UTC), and `extra`.

**Normalization formats chosen:**
| Field | Format |
|---|---|
| Phone | E.164 via `phonenumbers` library |
| Date | `datetime.date` via `python-dateutil`; "Present"/"Current" → `is_current=True` |
| Email | Lowercased, stripped; primary flagged on first occurrence |
| Skill | Alias dict (js→JavaScript, etc.) + title-case fallback |
| Location | Comma-split into city / state / country; state_code if ≤ 3 chars |
| Company | Suffix stripping (Inc, LLC, Ltd, Corp) via regex |
| Education | Degree → DegreeLevel enum (BACHELOR, MASTER, DOCTORATE, etc.) |

---

## Merge and Conflict-Resolution Policy

**Match keys** used for deduplication:
- Email: lowercased address
- Phone: E.164 normalised form (falls back to digit-only strip)
- Skill: lowercased canonical name
- Experience: `company_key | title_key | start_date`
- Education: `institution_key | degree_key | field_key`

**Merge behaviour:** All sources that agree on a value accumulate their `Provenance`
records into one list. Sources that disagree are stored in `ConfidenceField.conflicts`
for the resolver to evaluate.

**Conflict resolution — 5 deterministic rules (applied in order):**
1. Structured source wins over unstructured (ATS, CSV, LinkedIn, GitHub > Resume, Recruiter Notes)
2. Higher confidence wins (after `ConfidenceEngine` has scored provenance records)
3. Most recent `extracted_at` timestamp wins
4. Longer normalised company name wins (for experience conflicts)
5. Lexicographic order breaks any remaining tie (ensures stability across runs)

No randomness. No calls to `datetime.now()`. Same inputs always produce the same output.

---

## Confidence Assignment

Formula per field:

```
source_confidence(p) = source_weight[p.source_type] × method_weight[p.method]
field_confidence     = clamp(max(source_votes) + agreement_bonus − conflict_penalty)
overall_confidence   = weighted_average(field_confidence[f] × field_importance[f])
```

**Source weights** (from `ConfidenceFormula` dataclass, injectable):
`ATS_JSON=0.95, CSV=0.90, LinkedIn=0.88, GitHub=0.82, Manual=0.80, Resume=0.80, RecruiterNotes=0.62`

**Method weights:** `API_RESPONSE=0.98, STRUCTURED_FIELD=0.95, MANUAL=0.90, REGEX=0.78, NLP_HEURISTIC=0.70, INFERRED=0.62`

**Bonuses / penalties:** +0.04 per additional agreeing source (capped at +0.12); −0.08 per
unresolved conflict. Inferred skills (from GitHub repo analysis) are hard-capped at 0.70.
GitHub skills matched across a non-GitHub source get an additional +0.10 bonus.

`overall_confidence` is a weighted average across populated fields, using `field_importance`
weights (`emails=1.00, experience=0.90, phones=0.85, skills=0.80`, etc.).

---

## Runtime-Configurable Projection

The `ProjectionEngine` reads a JSON config at runtime and reshapes the `CanonicalCandidate`
into any output structure without code changes. Config format:

```json
{
  "fields": [
    { "path": "candidate_name", "from": "display_name" },
    { "path": "contact_email",  "from": "primary_email" },
    { "path": "phone_e164",     "from": "primary_phone", "normalize": "e164" },
    { "path": "skills",         "from": "skills",
      "array": { "fields": [{ "path": "name" }, { "path": "confidence" }] } }
  ],
  "on_missing": "omit",
  "include_provenance": false,
  "include_confidence": true
}
```

Supported config options: field selection, output key renaming (`path` vs `from`),
per-field normalization (`e164`, `iso3166`, `canonical`), type coercion (`string`,
`integer`, `float`, `boolean`, `array`), nested object and array descriptors, confidence
range remapping, and three missing-value strategies: `null` (include as null), `omit`
(exclude the key entirely), `error` (fail the projection with a structured error).

The canonical model and projection layer are fully decoupled. `ProjectionEngine.project()`
accepts any serializable object — it does not import `CanonicalCandidate`.

Output is validated against a JSON Schema (Draft 2020-12) via `JSONSchemaValidator`.

---

## Edge Cases Handled

1. **Malformed or missing source** — Adapters never crash. Missing fields → `None` +
   `ExtractionWarning`. If all sources fail extraction, the pipeline returns an empty
   list rather than a crash.

2. **Conflicting names across sources** — Resume says "Janet", ATS says "Jane". Both
   are preserved in `ConfidenceField.conflicts`. ConflictResolver picks ATS (structured
   source wins). The losing value remains auditable.

3. **Phone number variants** — `(415) 555-2671`, `+1 415 555 2671`, and `415.555.2671`
   all normalise to `+14155552671` and deduplicate to one `CanonicalPhone` entry.

4. **GitHub-only candidate** — GitHub bio provides name and location but no email.
   `candidate_id` falls back to the profile URL for UUID5 generation. Name confidence
   is reduced (GitHub names are unverified display names; name fields filter out GitHub
   provenance before scoring to avoid inflating confidence).

5. **CSV with multiple rows per candidate** — `extract_rows()` groups rows by a `uid`
   column and aggregates education, skill, and project records across rows into one
   `ExtractedCandidate`. Missing `uid` → row index used as fallback key.

---

## Deliberate Omissions (Time Pressure)

- **`years_experience` field** — not computed. The merge engine has the data
  (`experience[].start_date`, `end_date`, `duration_months`) but the aggregation
  step was not implemented.

- **Fuzzy skill matching** — alias dictionary covers ~15 common variants. A
  production system would use vector similarity or a curated taxonomy with thousands
  of entries. `rapidfuzz` is listed in requirements but not wired in.

- **Live LinkedIn extraction** — LinkedIn ToS prohibits scraping. The `LinkedInAdapter`
  handles an exported profile dictionary. The Streamlit UI passes a mock dict for
  demonstration; a production integration would accept a real LinkedIn export file.

- **CLI entrypoint** — `typer` is in requirements; no CLI was built. The Streamlit UI
  covers the "minimal UI" requirement.

- **Async / batch processing** — all I/O is synchronous. The architecture supports
  adding a worker pool (each adapter is stateless and independently callable), but
  this was not implemented.
