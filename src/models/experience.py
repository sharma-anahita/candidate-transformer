"""
experience.py — Work experience model.

Design rationale:
  Work experience is the highest-signal section of a candidate profile, but
  also the most structurally inconsistent across sources:
    - Resumes use free-text bullets under job entries
    - LinkedIn returns structured role objects with start/end month-year
    - ATS CSV exports flatten experience into "current company" + "current title"
    - GitHub infers experience from org membership and repo metadata

  To accommodate all sources, we store both raw date strings (``raw_start_date``,
  ``raw_end_date``) and parsed date objects (``start_date``, ``end_date``).
  Adapters store raw; the date normaliser parses them. This decoupling means
  a bug in date parsing only affects the normaliser, not the extractor.

  ``responsibilities`` and ``achievements`` are separated because achievements
  contain quantified outcomes ("Reduced API latency by 40%") which are high-value
  signals for ATS ranking and resume screening AI. Keeping them separate avoids
  burying them inside unstructured description blobs.

  ``technologies`` per role provides temporal context for skills: if a candidate
  used React in their 2019 role but not since, that affects the skill's
  ``last_used_year`` and confidence weight.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Experience(BaseModel):
    """
    A single work experience entry.

    Fields
    ------
    company:
        Organisation name, exactly as extracted. The company normaliser will
        resolve abbreviations and aliases later (e.g., "Google LLC" → "Google",
        "MSFT" → "Microsoft").

    title:
        Job title as listed. Not normalised — titles are so varied ("Staff SWE",
        "Senior Software Engineer L5", "Principal Engineer") that normalisation
        is a specialised NLP problem outside this pipeline's scope. Stored raw
        for downstream text-matching.

    start_date:
        Parsed start date. None when the adapter could not parse the raw string
        (malformed format, ambiguous year-only dates, etc.).

    raw_start_date:
        Original start date string from the source.
        E.g., "Jan 2020", "January 2020", "2020-01", "01/2020".
        Always stored so the date normaliser can retry with better heuristics
        or context hints (e.g., knowing end_date makes "Jan" unambiguous).

    end_date:
        Parsed end date. None if this is a current role OR if parsing failed.
        Disambiguated by ``is_current``: if is_current=True and end_date=None,
        the role is ongoing. If is_current=False and end_date=None, parsing
        failed and the normaliser should retry.

    raw_end_date:
        Original end date string. Commonly "Present", "Current", "now", or
        a formatted date. The model validator auto-sets ``is_current=True``
        when it detects "present"-like values here.

    is_current:
        True if this is the candidate's active role. Derived from raw_end_date
        or set explicitly by the adapter.
        Critical for salary/comp benchmarking and recency scoring.

    duration_months:
        Computed automatically from start_date and end_date (or today's date
        for current roles). Stored explicitly to avoid recomputing on every
        read and to preserve the value even when dates are later modified.
        None if start_date is unavailable.

    location:
        Office location for this specific role, stored as a raw string.
        Not a Location object — experience locations are rarely structured
        and are lower priority than the candidate's current location.
        The normaliser may optionally parse this into a Location.

    description:
        Full, unmodified description block from the source.
        Preserved for NLP downstream (LLM-based extraction, resume screening).

    responsibilities:
        Structured list of duties extracted from the description.
        Populated by the resume adapter (split on bullet points/line breaks).

    achievements:
        Quantified accomplishments from the description.
        High-value ATS signal — these are the lines with numbers, percentages,
        and outcomes. Extracted separately from generic responsibilities.

    technologies:
        Technologies mentioned in the context of this role.
        Feeds the temporal skill graph in the merge engine, which computes
        ``last_used_year`` and ``years_of_experience`` per skill.

    employment_type:
        "full-time", "part-time", "contract", "internship", "freelance".
        Stored as a plain string since vocabulary varies wildly across sources.
        The normaliser may standardise this to a controlled vocabulary.
    """

    company: str
    title: str
    start_date: Optional[date] = None
    raw_start_date: Optional[str] = None
    end_date: Optional[date] = None
    raw_end_date: Optional[str] = None
    is_current: bool = False
    duration_months: Optional[int] = None
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    employment_type: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "Experience":
        """
        Derive ``is_current`` and ``duration_months`` from available data.

        Runs after all field validators to ensure start_date and raw_end_date
        are in their final form before we compute derived values.

        is_current: inferred from raw_end_date containing "present"-like text.
        Explicit is_current=True always wins over this inference.

        duration_months: computed from start/end dates. For current roles,
        uses today's date as the end. Returns 0 for roles where end < start
        (malformed data, e.g., a resume with wrong years) rather than raising.
        """
        # Step 1: infer is_current from raw_end_date
        if not self.is_current and self.raw_end_date:
            raw = self.raw_end_date.strip().lower()
            if raw in {"present", "current", "now", "ongoing", "—", "–", "-", ""}:
                self.is_current = True

        # Step 2: compute duration_months from start/end dates
        if self.duration_months is None and self.start_date is not None:
            end = self.end_date
            if end is None and self.is_current:
                end = date.today()

            if end is not None:
                months = (
                    (end.year - self.start_date.year) * 12
                    + (end.month - self.start_date.month)
                )
                self.duration_months = max(0, months)

        return self
