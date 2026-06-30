"""
provenance.py — The lineage and confidence layer of the pipeline.

Every field in the canonical profile traces back to one or more Provenance
records. This module defines:

  - SourceType        : enum of all supported input source types
  - ExtractionMethod  : enum of all ways a value can be obtained
  - Provenance        : a single lineage record for one field from one source
  - ConfidenceField   : generic wrapper binding a canonical value to its
                        confidence score and provenance chain

Design rationale:
  Provenance is NOT an afterthought. It is the data contract that enables:
    1. GDPR right-of-access: "Where does this data come from?"
    2. Confidence scoring: API fields > regex-extracted > inferred
    3. Merge arbitration: choose the highest-confidence value among conflicts
    4. Cache invalidation: re-fetch only stale sources (compare extracted_at)
    5. Debuggability: raw_value lets you trace normalization bugs backwards

  ConfidenceField[T] uses Python Generics so the type system enforces that
  ConfidenceField[str] cannot be assigned to ConfidenceField[int], etc.
  Pydantic v2 natively supports Generic BaseModels with full serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class SourceType(str, Enum):
    """
    All supported input source types.

    Using ``str, Enum`` so values serialise to plain strings in JSON output.
    This matters for the projection layer, which compares source types against
    runtime config keys without importing this enum.

    Ordering is irrelevant; weights are defined in the confidence engine config.
    """

    RESUME = "resume"
    CSV = "csv"
    ATS_JSON = "ats_json"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    RECRUITER_NOTES = "recruiter_notes"
    MANUAL = "manual"


class ExtractionMethod(str, Enum):
    """
    Describes *how* a field value was obtained.

    The confidence engine uses this to apply a reliability multiplier:
      API_RESPONSE > STRUCTURED_FIELD > REGEX > NLP_HEURISTIC > INFERRED > MANUAL

    Having a dedicated enum (rather than a free-form string) means confidence
    weights can be looked up in a dict without string-matching bugs.
    """

    # Data came directly from a typed API response (e.g., GitHub REST API).
    # Highest reliability — the platform itself provided this value.
    API_RESPONSE = "api_response"

    # Mapped from a structured, schema'd file (CSV column, JSON key).
    # Very reliable — field boundaries are explicit.
    STRUCTURED_FIELD = "structured_field"

    # Extracted via regular expression from unstructured text.
    # Moderate reliability — depends on regex quality and text formatting.
    REGEX = "regex"

    # Extracted via a rule-based or statistical NLP heuristic.
    # Lower reliability — heuristics degrade on edge cases.
    NLP_HEURISTIC = "nlp_heuristic"

    # Derived/inferred from other data (e.g., skills inferred from repo files).
    # Lowest reliability — never directly stated by the candidate.
    INFERRED = "inferred"

    # Entered by a human (recruiter, hiring manager, etc.).
    # Reliability depends on context; treated as STRUCTURED_FIELD by default.
    MANUAL = "manual"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────


class Provenance(BaseModel):
    """
    A single lineage record for one field value from one source.

    Multiple Provenance records are collected into a list on ConfidenceField
    when a value was seen across several sources (e.g., name extracted from
    both resume and LinkedIn). This multi-source list is what the confidence
    engine uses to compute cross-source agreement bonuses.

    Fields
    ------
    source_type:
        Which category of source this came from (resume, github, …).
        Used for source-weight lookups in the confidence engine.

    adapter_name:
        The fully qualified Python class name of the adapter that ran.
        E.g., ``"ResumeAdapter"``, ``"GitHubAdapter"``.
        Stored as a plain string so this model stays decoupled from adapter
        implementations — no circular imports.

    method:
        How the value was derived. Drives the confidence engine's reliability
        multiplier (API beats regex beats inference).

    source_id:
        Opaque identifier for the specific source artifact: a file path, a
        GitHub URL, a database row ID, etc. Enables cache lookups and linking
        provenance records to original documents for audit responses.

    raw_value:
        The original, un-normalised string exactly as it appeared in the source.
        Indispensable for debugging normalisation bugs: if a phone number is
        wrong in the canonical profile, you can trace it back to the raw string
        and identify whether the extractor or normaliser is at fault.

    confidence:
        The extractor's local confidence at extraction time. This is NOT the
        same as the final canonical confidence — that is computed later by the
        confidence engine from multiple provenance records. Think of this as the
        per-source "vote weight".

    extracted_at:
        UTC timestamp of when this extraction event occurred. Used for:
          - Freshness scoring (more recent = higher confidence for time-sensitive
            fields like job title or location).
          - Cache invalidation: if source_id was fetched more recently elsewhere,
            discard this stale provenance.

    extra:
        Arbitrary adapter-specific metadata that doesn't fit the canonical schema.
        Examples:
          - GitHub adapter: ``{"stars": 1200, "forks": 80, "language": "Python"}``
          - ATS adapter: ``{"ats_field_name": "applicant_email", "row_index": 3}``
        Not used by the core pipeline but preserved for domain extensions and
        debugging.

    Notes
    -----
    Provenance is frozen (immutable) once created. A lineage record must never
    be silently mutated — that would undermine the entire audit trail.
    """

    source_type: SourceType
    adapter_name: str
    method: ExtractionMethod
    source_id: str
    raw_value: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceField — Generic wrapper for canonical field values
# ─────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")


class ConfidenceField(BaseModel, Generic[T]):
    """
    Wraps a canonical field value with its confidence score and full
    provenance chain.

    This is the atomic unit of the CanonicalCandidate. Every scalar field
    in the canonical profile (first_name, summary, location, …) is typed as
    ``Optional[ConfidenceField[T]]``.

    Why wrap every field instead of a single top-level provenance dict?
    ──────────────────────────────────────────────────────────────────────
    A flat provenance dict keyed by field name loses structure: you'd need
    ``provenance["first_name"][0].confidence`` instead of accessing it
    directly on the field object. Wrapping each field:
      1. Keeps confidence and provenance co-located with the value — no
         possibility of them drifting out of sync.
      2. Lets the Projection Engine decide at runtime whether to include or
         strip confidence/provenance per-field based on the output config.
      3. Enables downstream consumers to implement their own confidence
         thresholds field-by-field without knowing the pipeline internals.

    Fields
    ------
    value:
        The normalized, canonical field value. Type-parameterised as T, so
        ``ConfidenceField[str]`` for names, ``ConfidenceField[float]`` for GPA,
        ``ConfidenceField[Location]`` for location, etc.

    confidence:
        Final confidence in [0.0, 1.0] after cross-source agreement,
        validation success, and source-weight blending by the confidence engine.
        This is distinct from any individual Provenance.confidence — it is the
        aggregate verdict.

    provenance:
        Ordered list of all Provenance records that contributed to this value.
        The first element is the "winning" source (highest confidence, used as
        the canonical value). Subsequent elements are supporting sources that
        agree, or losing sources that were overridden.

    is_inferred:
        True when the value was derived/computed rather than directly stated.
        Inferred fields have a hard confidence ceiling defined in the confidence
        engine config — you can never be fully confident about an inference.

    conflicts:
        Values that appeared in other sources but lost conflict resolution.
        Stored verbatim for auditability. E.g., if GitHub says location is
        "San Francisco" and LinkedIn says "Remote", and we pick "San Francisco",
        the conflict list records {"source": "linkedin", "value": "Remote"}.

    Notes
    -----
    ``arbitrary_types_allowed = True`` is required because T can be any type,
    including non-Pydantic types like ``date`` or ``UUID``.
    """

    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)
    is_inferred: bool = False
    conflicts: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
