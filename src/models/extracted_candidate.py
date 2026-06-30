"""
extracted_candidate.py — Raw output of every adapter.

This is the single contract all adapters fulfil. Every adapter — regardless
of source type — returns exactly one ExtractedCandidate. The pipeline never
needs to know which adapter produced it; it only interacts with this model.

Architecture rules enforced here:
  - ALL fields are Optional or have empty defaults. Adapters must never fail
    because a field is missing. Missing data = None, not an exception.
  - ExtractionWarning stores non-fatal issues instead of raising. This enables
    partial profiles (e.g., a PDF where OCR failed on the phone number) to
    flow through the pipeline rather than blocking all processing.
  - ``raw_text`` preserves the source text for text-based sources (resumes,
    recruiter notes) so re-processing is possible without re-fetching.
  - ``metadata`` captures adapter-specific data that doesn't fit the canonical
    schema. Nothing is thrown away — it just gets downgraded to metadata.

Design decision: flat vs nested identity fields
  We store first_name, middle_name, last_name AND full_name separately.
  Some sources provide structured name (ATS JSON: first_name + last_name).
  Others provide only the full string (resume headline, GitHub display name).
  The name normaliser handles splitting full_name when parts are absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.education import Education
from src.models.email import Email
from src.models.experience import Experience
from src.models.location import Location
from src.models.phone import Phone
from src.models.profile import Profile
from src.models.provenance import SourceType
from src.models.skill import Skill


# ─────────────────────────────────────────────────────────────────────────────
# ExtractionWarning
# ─────────────────────────────────────────────────────────────────────────────


class ExtractionWarning(BaseModel):
    """
    A non-fatal issue encountered during adapter extraction.

    Adapters must NEVER raise on bad data — they store warnings instead.
    This is a hard architectural rule: one bad field should never crash
    the pipeline for the remaining valid fields.

    Fields
    ------
    field:
        Canonical name of the field where the issue occurred.
        E.g., "phone", "start_date", "gpa". Used for grouping in the UI.

    message:
        Human-readable description of the problem.
        E.g., "Could not parse phone: '555-CALL-NOW' is not a valid number."

    raw:
        The raw value that caused the issue. Stored for debugging.
        E.g., "555-CALL-NOW". None when the issue is structural
        (e.g., "required column 'email' not found in CSV").
    """

    field: str
    message: str
    raw: Optional[str] = None

    model_config = ConfigDict(frozen=True)


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedCandidate
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedCandidate(BaseModel):
    """
    Raw, unnormalised output of a single adapter run against a single source.

    Invariants (enforced by convention, not code — see BaseAdapter):
      1. Produced exclusively by adapters.
      2. Values are raw, unnormalised. E.g., phones are stored as extracted
         strings; dates may be "Jan 2020" not a date object.
      3. Never modified after construction (adapters build it, pipeline reads it).
      4. One ExtractedCandidate per source document / API response.

    Fields
    ------
    extraction_id:
        UUID generated at construction time. Unique per extraction event.
        Distinct from candidate_id (which identifies the person) — this
        identifies the specific run of an adapter against a specific source.
        Used for caching, deduplication of re-runs, and linking warnings to events.

    source_type:
        Category of the source this came from. Set by the adapter.

    source_id:
        Opaque identifier for the specific source artifact.
        Examples: "/path/to/resume.pdf", "https://github.com/janedoe",
        "greenhouse:applicant:12345", "csv:row:7".
        Used for cache lookups and provenance linking.

    adapter_name:
        Python class name of the adapter that produced this.
        Stored as a plain string to avoid coupling this model to adapter code.
        E.g., "CSVAdapter", "GitHubAdapter".

    extracted_at:
        UTC timestamp of when this extraction occurred.
        Used for freshness scoring and cache invalidation.

    raw_text:
        Full raw text content for text-based sources.
        Stored for resume parsing (full PDF text), recruiter notes, etc.
        None for structured sources (CSV, ATS JSON) where there is no
        monolithic text blob.

    --- Identity ---
    first_name, middle_name, last_name:
        Name components when the source provides them separately.

    full_name:
        Full name when the source provides it as one string (common in
        GitHub profiles, LinkedIn display names). The name normaliser will
        attempt to split this into parts when the component fields are absent.

    --- Contact ---
    emails:
        All email addresses found in this source, in order of appearance.

    phones:
        All phone numbers found, with raw value always set and normalised
        value populated only by the phone normaliser (not the adapter).

    location:
        Best-effort location from this source. May be a raw string with
        only the ``raw`` field populated.

    --- Professional ---
    summary:
        Professional summary, bio, or objective statement, as extracted.

    skills:
        Skills extracted from this source. Category and proficiency are
        set to UNKNOWN at extraction time — the skill normaliser sets them.

    experience:
        Work history entries. Dates are stored as raw strings (raw_start_date,
        raw_end_date) at extraction time; parsed date objects are populated
        by the date normaliser.

    education:
        Academic history. Same dual-date pattern as experience.

    profiles:
        Social and professional profile links found in this source.

    --- Adapter overflow ---
    metadata:
        Adapter-specific data that doesn't fit the canonical schema.
        E.g., GitHub: {"stars": 1200, "followers": 450, "orgs": ["torvalds"]}.
        Not used by core pipeline logic but preserved for downstream extensions
        and audit logging.

    warnings:
        Non-fatal extraction issues, appended via ``add_warning()``.
        Pipeline continues processing regardless of warnings.
    """

    # Extraction metadata
    extraction_id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    source_id: str
    adapter_name: str
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw_text: Optional[str] = None

    # Identity
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None

    # Contact
    emails: list[Email] = Field(default_factory=list)
    phones: list[Phone] = Field(default_factory=list)
    location: Optional[Location] = None

    # Professional
    summary: Optional[str] = None
    skills: list[Skill] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    profiles: list[Profile] = Field(default_factory=list)

    # Overflow and diagnostics
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ExtractionWarning] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)

    # ── Convenience methods ───────────────────────────────────────────────────

    def add_warning(
        self,
        field: str,
        message: str,
        raw: Optional[str] = None,
    ) -> None:
        """
        Append a non-fatal extraction warning.

        Adapters call this instead of raising exceptions when encountering
        bad but non-blocking data. The pipeline continues; the warning is
        surfaced in the UI and stored in the provenance audit trail.

        Args:
            field:   Canonical field name where the issue occurred.
            message: Human-readable description of the problem.
            raw:     The raw value that caused the issue, if applicable.
        """
        self.warnings.append(
            ExtractionWarning(field=field, message=message, raw=raw)
        )

    @property
    def has_warnings(self) -> bool:
        """True if any non-fatal extraction warnings were recorded."""
        return len(self.warnings) > 0

    @property
    def display_name(self) -> Optional[str]:
        """
        Best-effort display name assembled from available name components.

        Prefers component fields (first + middle + last) over full_name
        to avoid double-storing the same string when the normaliser has
        already split full_name into components.
        """
        parts = [
            p
            for p in [self.first_name, self.middle_name, self.last_name]
            if p
        ]
        if parts:
            return " ".join(parts)
        return self.full_name
