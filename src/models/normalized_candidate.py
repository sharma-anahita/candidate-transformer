"""
normalized_candidate.py — Post-normalisation candidate data.

After adapters extract raw data into ExtractedCandidate objects, each one
passes through the normalisation pipeline. The output is a NormalizedCandidate
— same structure, but with clean, canonical values:

  - phones: ``normalized`` field populated in E.164 format (+12125551234)
  - emails: lowercased, validated, ``domain`` set
  - dates: ``start_date``/``end_date`` parsed from raw strings into date objects
  - skills: ``name`` mapped to canonical taxonomy entry
  - names: properly title-cased, ``full_name`` split into components if needed
  - location: ``country_code`` set, ``city``/``state`` parsed from raw string

Design decision: separate model vs. mutating ExtractedCandidate
──────────────────────────────────────────────────────────────────
We do NOT mutate ExtractedCandidate in place. A separate NormalizedCandidate:
  1. Preserves the original extracted data for debugging and audit
  2. Makes the normalisation step explicit in the pipeline (clear input/output)
  3. Allows re-normalisation when normaliser logic changes, without re-extraction
  4. Enables A/B testing normaliser versions by comparing their outputs

NormalizationLog records every normalisation applied: what normaliser ran,
what the value was before, what it is after, and whether it succeeded.
This trace is invaluable for debugging "why is my phone number wrong?" tickets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

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
# NormalizationLog
# ─────────────────────────────────────────────────────────────────────────────


class NormalizationLog(BaseModel):
    """
    A single normalisation event applied to one field.

    One NormalizedCandidate accumulates many NormalizationLog entries —
    one per field per normaliser that ran. This provides a complete audit
    trail of all transformations applied to a candidate's data.

    Fields
    ------
    field:
        Canonical field name that was normalised.
        E.g., "phones[0].normalized", "emails[0].address", "first_name".

    normalizer:
        Python class name of the normaliser that ran.
        E.g., "PhoneNormalizer", "EmailNormalizer", "SkillNormalizer".

    original_value:
        Value before normalisation, always as a string representation.
        None for fields where there was no pre-existing value.

    normalized_value:
        Value after normalisation, as a string representation.
        None if normalisation failed or produced no output.

    success:
        True if normalisation produced a valid result.
        False if the normaliser could not parse/map the value — in which
        case the field is left with whatever value the adapter set.

    message:
        Human-readable explanation when success=False (why it failed)
        or when a notable transformation occurred (e.g., "Mapped 'js' → 'JavaScript'
        via skill taxonomy").
    """

    field: str
    normalizer: str
    original_value: Optional[str] = None
    normalized_value: Optional[str] = None
    success: bool = True
    message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCandidate
# ─────────────────────────────────────────────────────────────────────────────


class NormalizedCandidate(BaseModel):
    """
    Candidate data after normalisation, ready for the merge engine.

    Structure mirrors ExtractedCandidate but values are clean:
      - Phone.normalized is set (E.164 format)
      - Email.address is lowercase with a valid format
      - Experience.start_date / end_date are date objects (not raw strings)
      - Skill.name is a canonical taxonomy entry
      - Location.country_code is set

    Fields
    ------
    extraction_id:
        UUID of the ExtractedCandidate this was derived from.
        Critical for lineage: the merge engine uses this to link a
        NormalizedCandidate back to its original source for provenance tracking.

    source_type:
        Preserved from the ExtractedCandidate.

    source_id:
        Preserved from the ExtractedCandidate.

    adapter_name:
        Preserved from the ExtractedCandidate.

    normalized_at:
        UTC timestamp of when normalisation ran. Used for cache invalidation:
        if the normaliser version changes, re-normalise all records with
        normalization_version < current version.

    normalization_logs:
        Complete ordered log of all normalisation events applied to this
        candidate. Append-only; indexed by field name for O(1) lookup in the UI.

    normalization_version:
        Semantic version of the normaliser configuration that produced this.
        When normaliser logic changes (new skill taxonomy, new phone regex),
        increment this version so stale NormalizedCandidate objects can be
        identified and re-processed without re-extraction.

    [All candidate data fields mirror ExtractedCandidate]
    """

    # Lineage
    extraction_id: UUID
    source_type: SourceType
    source_id: str
    adapter_name: str
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    normalization_logs: list[NormalizationLog] = Field(default_factory=list)
    normalization_version: str = "1.0.0"

    # Identity (normalised: title-cased, full_name split if needed)
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None

    # Contact (normalised: phones E.164, emails lowercase + validated)
    emails: list[Email] = Field(default_factory=list)
    phones: list[Phone] = Field(default_factory=list)
    location: Optional[Location] = None

    # Professional (normalised: dates parsed, skills canonical)
    summary: Optional[str] = None
    skills: list[Skill] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    profiles: list[Profile] = Field(default_factory=list)

    # Adapter-specific overflow (preserved as-is from extraction)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)

    # ── Convenience methods ───────────────────────────────────────────────────

    def log(
        self,
        field: str,
        normalizer: str,
        original: Optional[str],
        result: Optional[str],
        success: bool = True,
        message: Optional[str] = None,
    ) -> None:
        """
        Record a single normalisation event.

        Called by each normaliser after processing a field.
        Appends a NormalizationLog entry for the audit trail.

        Args:
            field:      Canonical field path (e.g., "phones[0].normalized").
            normalizer: Normaliser class name (e.g., "PhoneNormalizer").
            original:   Value before normalisation (as string).
            result:     Value after normalisation (as string), or None on failure.
            success:    Whether normalisation produced a valid result.
            message:    Optional explanation or warning message.
        """
        self.normalization_logs.append(
            NormalizationLog(
                field=field,
                normalizer=normalizer,
                original_value=original,
                normalized_value=result,
                success=success,
                message=message,
            )
        )

    @property
    def display_name(self) -> Optional[str]:
        """Best-effort display name from available name components."""
        parts = [p for p in [self.first_name, self.middle_name, self.last_name] if p]
        return " ".join(parts) if parts else self.full_name

    @property
    def failed_normalizations(self) -> list[NormalizationLog]:
        """All normalisation events where success=False."""
        return [log for log in self.normalization_logs if not log.success]
