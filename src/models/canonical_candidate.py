"""
canonical_candidate.py — The final merged, confidence-scored candidate profile.

This is the central output of the pipeline. Every field carries its confidence
score and provenance chain. The Projection Engine reads this and reshapes it
into any client-specific output schema at runtime.

Key design decisions:
  1. Canonical*  submodels (CanonicalSkill, CanonicalEmail, etc.) extend their
     base counterparts by adding ``confidence`` and ``provenance`` fields. This
     avoids redefining all fields while making the canonical-specific additions
     explicit. Inheritance here is justified: CanonicalSkill IS-A Skill.

  2. Scalar identity fields (first_name, last_name, etc.) are wrapped in
     Optional[ConfidenceField[str]]. The Optional models the case where no
     source provided this field at all. The ConfidenceField wraps the value
     with its confidence and provenance when a source did provide it.

  3. List fields (skills, experience, education) use Canonical* submodels
     that carry provenance per item. This is more granular than a top-level
     provenance dict, which would lose the item-to-source mapping.

  4. ``needs_review()`` is a pipeline hook: any system consuming this profile
     can check if human review is needed without knowledge of the internals.

  5. The canonical profile is produced by the merge engine and thereafter
     treated as a value object — read-only from the perspective of all
     downstream consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.education import Education
from src.models.email import Email
from src.models.experience import Experience
from src.models.location import Location
from src.models.phone import Phone
from src.models.profile import Profile
from src.models.provenance import ConfidenceField, Provenance
from src.models.skill import Skill


# ─────────────────────────────────────────────────────────────────────────────
# Canonical sub-models
# Each adds confidence + provenance to its base model.
# ─────────────────────────────────────────────────────────────────────────────


class CanonicalEmail(Email):
    """
    Email with canonical confidence and provenance.

    Inherits all Email fields and validators. Adds:
      confidence  — how much to trust this email address.
      provenance  — which source(s) provided this address.
    """

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


class CanonicalPhone(Phone):
    """Phone with canonical confidence and provenance."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


class CanonicalSkill(Skill):
    """
    Skill with canonical confidence and provenance.

    ``confidence`` reflects how certain we are that the candidate has this
    skill. Explicitly listed skills start at higher confidence than inferred
    ones, and cross-source agreement raises confidence further.
    """

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


class CanonicalExperience(Experience):
    """
    Experience with canonical confidence and provenance.

    ``confidence`` reflects data quality: a fully-structured ATS record
    with verified company name scores higher than a PDF-parsed entry
    with unresolved company abbreviations.
    """

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


class CanonicalEducation(Education):
    """Education with canonical confidence and provenance."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


class CanonicalProfile(Profile):
    """Profile with canonical confidence and provenance."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CanonicalCandidate
# ─────────────────────────────────────────────────────────────────────────────


class CanonicalCandidate(BaseModel):
    """
    The fully merged, normalised, confidence-scored candidate profile.

    This is the single source of truth for a candidate across all input sources.
    Once produced by the merge engine, it is consumed by:
      - The Projection Engine (to produce client-specific JSON)
      - The UI (to display the candidate profile with confidence indicators)
      - Downstream ATS/ML systems (for matching, ranking, routing)

    Fields
    ------
    candidate_id:
        Stable UUID identifying this candidate across all pipeline runs.
        Different from extraction_id (which identifies a single source document
        extraction event). Two pipeline runs for the same candidate produce the
        same candidate_id (linked by email or other unique identifiers).

    merged_from:
        List of extraction_ids (from ExtractedCandidate) that were merged
        to produce this canonical profile. Full lineage from sources to
        canonical form.

    created_at:
        UTC timestamp of when this canonical profile was first created.

    updated_at:
        UTC timestamp of the most recent merge or enrichment update.

    pipeline_version:
        Semantic version of the pipeline that produced this. Enables
        identifying canonical profiles that need re-processing when the
        pipeline logic changes.

    --- Identity ---
    first_name, middle_name, last_name:
        Wrapped in ConfidenceField[str] so confidence and provenance are
        co-located with the value. Optional at the outer level because
        the field may have been completely absent across all sources.

    --- Contact ---
    emails:
        All unique email addresses, deduplicated across sources.
        The is_primary flag on CanonicalEmail marks the outreach address.

    phones:
        All unique phone numbers in E.164 format.

    location:
        ConfidenceField[Location] — the confidence wraps the entire location
        object because cross-source agreement on location is rare; when it
        occurs, it is strong signal.

    --- Professional ---
    summary:
        The selected or synthesised professional summary.
        Wrapped in ConfidenceField[str] with provenance indicating which
        source the selected summary came from.

    skills:
        Deduplicated, confidence-scored skill list. Ordered by confidence
        descending by default, but the projection engine can reorder.

    experience:
        Deduplicated work history, ordered by start_date descending.

    education:
        Deduplicated academic history, ordered by end_date descending.

    profiles:
        Deduplicated social/professional profile links.

    --- Scoring ---
    overall_confidence:
        Aggregate confidence in [0.0, 1.0], computed by the confidence engine
        as a weighted average of key field confidences (name, email, phone,
        skills) with field-importance weights from the pipeline config.
        This is the primary signal for routing to human review.

    field_confidences:
        Per-field confidence breakdown keyed by field name.
        E.g., {"first_name": 0.95, "phone": 0.72, "location": 0.60}.
        Stored flat for O(1) UI lookup without traversing all ConfidenceFields.

    --- Audit ---
    provenance:
        Complete provenance map keyed by field name.
        E.g., {"first_name": [Provenance(...), Provenance(...)], ...}.
        Used for GDPR right-of-access responses: every data point can be
        traced to its exact origin.
    """

    candidate_id: UUID = Field(default_factory=uuid4)
    merged_from: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "1.0.0"

    # Identity
    first_name: Optional[ConfidenceField[str]] = None
    middle_name: Optional[ConfidenceField[str]] = None
    last_name: Optional[ConfidenceField[str]] = None

    # Contact
    emails: list[CanonicalEmail] = Field(default_factory=list)
    phones: list[CanonicalPhone] = Field(default_factory=list)
    location: Optional[ConfidenceField[Location]] = None

    # Professional
    summary: Optional[ConfidenceField[str]] = None
    skills: list[CanonicalSkill] = Field(default_factory=list)
    experience: list[CanonicalExperience] = Field(default_factory=list)
    projects: list[CanonicalExperience] = Field(default_factory=list)
    education: list[CanonicalEducation] = Field(default_factory=list)
    profiles: list[CanonicalProfile] = Field(default_factory=list)

    # Scoring
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    field_confidences: dict[str, float] = Field(default_factory=dict)

    # Audit
    provenance: dict[str, list[Provenance]] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def display_name(self) -> Optional[str]:
        """Assemble display name from confidence-wrapped name fields."""
        parts = [
            cf.value
            for cf in [self.first_name, self.middle_name, self.last_name]
            if cf is not None
        ]
        return " ".join(parts) if parts else None

    @property
    def primary_email(self) -> Optional[str]:
        """
        Return the primary email address string.

        First looks for an email explicitly marked is_primary=True.
        Falls back to the first email in the list.
        """
        for email in self.emails:
            if email.is_primary:
                return email.address
        return self.emails[0].address if self.emails else None

    @property
    def primary_phone(self) -> Optional[str]:
        """
        Return the primary phone number in E.164 format.

        Prefers the normalized E.164 form. Falls back to the raw string
        if normalisation failed. Returns None if no phones present.
        """
        for phone in self.phones:
            if phone.is_primary:
                return phone.normalized or phone.raw
        if self.phones:
            return self.phones[0].normalized or self.phones[0].raw
        return None

    def needs_review(self, threshold: float = 0.70) -> bool:
        """
        Return True if the overall confidence is below ``threshold``.

        Used by the pipeline orchestrator to route low-confidence profiles
        to a human review queue rather than auto-publishing them to the ATS.

        Args:
            threshold: Minimum confidence to consider the profile auto-approved.
                       Default 0.70 — below this, human review is required.

        Returns:
            True if overall_confidence < threshold.
        """
        return self.overall_confidence < threshold

    def get_skill_names(self) -> list[str]:
        """Return all canonical skill names in confidence-descending order."""
        return [
            s.name
            for s in sorted(self.skills, key=lambda x: x.confidence, reverse=True)
        ]
