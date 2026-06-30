"""
education.py — Education history model.

Design rationale:
  Education data has two common sources:
    1. Resume/PDF — free-text with inconsistent formatting
    2. ATS structured fields — usually just institution + degree + year

  The dual raw/parsed date pattern (same as Experience) is applied here.
  GPA is particularly tricky because the scale varies by country and school
  (US: 4.0, India: 10.0, Germany: 1–6 inverted). Storing ``gpa_scale``
  alongside ``gpa`` prevents meaningless comparisons (a 3.8/4.0 is not
  worse than a 9.2/10.0 — they're equivalent at ~95th percentile).

  ``DegreeLevel`` enables structured filtering ("must have Bachelor's or above")
  without brittle string matching against 50 variants of "Bachelor of Science".

  ``courses`` matters for early-career candidates who lack experience but have
  relevant academic background — e.g., a new grad listing "Distributed Systems"
  and "Machine Learning" is signalling what a job description requires.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class DegreeLevel(str, Enum):
    """
    Normalised academic credential level.

    Enables ATS filter queries like "minimum Bachelor's degree" without
    matching against free-text like "BS", "B.Sc", "Bachelor of Science",
    "Licenciatura" (Spanish equivalent), or "Licence" (French equivalent).

    BOOTCAMP and ONLINE_COURSE are included because they carry real signal
    in tech hiring — a Coursera ML Specialisation is meaningful for
    data-science roles even without a formal degree.
    """

    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    CERTIFICATE = "certificate"
    BOOTCAMP = "bootcamp"
    ONLINE_COURSE = "online_course"
    OTHER = "other"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Education model
# ─────────────────────────────────────────────────────────────────────────────


class Education(BaseModel):
    """
    A single academic credential or training record.

    Fields
    ------
    institution:
        School, university, or training provider name, as extracted.
        The education normaliser resolves abbreviations and alternate names
        (e.g., "MIT" → "Massachusetts Institute of Technology",
        "UC Berkeley" → "University of California, Berkeley").

    degree:
        Degree name as listed in the source (e.g., "B.S. in Computer Science",
        "Bachelor of Engineering").  Stored raw; degree_level is the normalised
        counterpart for machine comparisons.

    degree_level:
        Normalised degree level enum. Set by the education normaliser by mapping
        the raw ``degree`` string against a lookup table.
        Defaults to UNKNOWN at extraction time.

    field_of_study:
        Academic major or discipline (e.g., "Computer Science", "Electrical
        Engineering", "Business Administration"). Extracted separately from
        ``degree`` because they're used in different ATS filter contexts.

    start_date / raw_start_date:
        Same dual-storage pattern as Experience — see experience.py for
        the full rationale.

    end_date / raw_end_date:
        raw_end_date often contains "Expected May 2025" for ongoing programs.
        The normaliser detects "Expected" or "Anticipated" prefixes and sets
        a flag (not implemented in this model) rather than treating it as a
        past graduation.

    gpa:
        Grade Point Average as a float. None when not reported.
        Stored alongside gpa_scale so comparisons are always contextualised.

    gpa_scale:
        The scale the GPA is measured on. Defaults to 4.0 (US standard).
        Adapters MUST set this correctly when extracting non-4.0-scale GPAs.
        The model validator enforces gpa ≤ gpa_scale to catch unit errors.

    honors:
        Latin honours or equivalent (e.g., "Summa Cum Laude", "First Class",
        "Distinction"). Not normalised at extraction time.

    activities:
        Clubs, sports teams, student societies. Soft-skill and culture-fit
        signal, especially relevant for early-career candidates.

    courses:
        Relevant coursework listed by the candidate.
        Feeds the skill graph for new-grad candidates who lack experience
        but have strong academic depth.

    is_verified:
        Whether a third-party service has confirmed this credential.
        Always False at extraction time. Set by an optional enrichment step.
    """

    institution: str
    degree: Optional[str] = None
    degree_level: DegreeLevel = DegreeLevel.UNKNOWN
    field_of_study: Optional[str] = None
    start_date: Optional[date] = None
    raw_start_date: Optional[str] = None
    end_date: Optional[date] = None
    raw_end_date: Optional[str] = None
    gpa: Optional[float] = Field(default=None, ge=0.0)
    gpa_scale: float = Field(default=4.0, gt=0.0)
    honors: Optional[str] = None
    activities: list[str] = Field(default_factory=list)
    courses: list[str] = Field(default_factory=list)
    is_verified: bool = False

    model_config = ConfigDict(populate_by_name=True)

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_gpa_against_scale(self) -> "Education":
        """
        Ensure GPA does not exceed the declared scale.

        This catches common extraction errors where gpa_scale was not set
        alongside gpa (e.g., adapter extracts gpa=9.2 with default scale 4.0).
        Raises ValueError so the adapter can catch it and store a warning
        instead of silently storing a corrupt education record.
        """
        if self.gpa is not None and self.gpa > self.gpa_scale:
            raise ValueError(
                f"GPA {self.gpa} exceeds the declared scale {self.gpa_scale}. "
                f"If this is a {self.gpa_scale:.0f}-scale institution, set "
                f"gpa_scale correctly (e.g., gpa_scale=10.0 for Indian universities)."
            )
        return self
