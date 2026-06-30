"""
email.py — Email address model.

Design rationale:
  Emails are deceptively complex. In the real world:
    - Candidates list work, personal, school, and old addresses
    - Addresses appear with inconsistent casing ("John@Gmail.COM")
    - ATS systems need to know which email to use for outreach
    - GDPR requires knowing *exactly* what email was extracted from *where*

  This model separates the raw address from metadata about its type and
  primary status. The normaliser populates ``domain`` automatically so
  downstream systems can do employer inference without re-parsing the string.

  We do NOT use Pydantic's built-in ``EmailStr`` because it applies RFC 5322
  strictness that real-world resumes violate (e.g., addresses with spaces
  or uncommon TLDs extracted by imperfect PDF parsers). Instead, we apply a
  permissive but reasonable regex and store the validated form.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A practical email regex — stricter than "anything with @" but looser than
# RFC 5322. Handles the vast majority of real-world email addresses.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class EmailType(str, Enum):
    """
    Classifies the role of this email address.

    ATS systems and recruiters need to distinguish work emails (often monitored
    by employers) from personal emails (where a passive candidate can be reached
    discreetly). School emails are flagged separately because they expire after
    graduation and should not be used for long-term outreach.
    """

    WORK = "work"
    PERSONAL = "personal"
    SCHOOL = "school"
    OTHER = "other"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Email model
# ─────────────────────────────────────────────────────────────────────────────


class Email(BaseModel):
    """
    A single email address with rich metadata.

    Fields
    ------
    address:
        The email address, always stored in lowercase after the field validator
        runs. We store only the normalised form — the raw form is preserved in
        the Provenance.raw_value attached at the CanonicalCandidate level.

        Why not store both raw and normalised here?  Because Email is used in
        three contexts (ExtractedCandidate, NormalizedCandidate, CanonicalEmail)
        and only the extractor context needs the raw form; preserving it in every
        context would waste memory and create confusion about which is authoritative.
        The raw value is always traceable via the Provenance chain.

    type:
        Work / personal / school / other.  Defaults to UNKNOWN when the adapter
        cannot determine type from context.  The normaliser may upgrade UNKNOWN
        to a more specific type based on domain heuristics (e.g., gmail.com →
        likely personal).

    is_primary:
        Flags the single email the pipeline has decided to use for outreach.
        Only one email per CanonicalCandidate should be primary. This flag is
        set by the merge engine, not by adapters — adapters cannot know which
        source is most trustworthy.

    is_verified:
        Whether deliverability has been confirmed (e.g., a verification ping).
        Always False at extraction time. May be set true by an optional
        verification step downstream.

    domain:
        The domain portion of the address (everything after ``@``).
        Derived automatically in the model validator if not supplied.
        Stored explicitly to avoid repeated string operations in queries and
        to enable employer-inference logic without parsing the address again.
        E.g., ``@google.com`` → candidate likely works at Google.
    """

    address: str
    type: EmailType = EmailType.UNKNOWN
    is_primary: bool = False
    is_verified: bool = False
    domain: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("address", mode="before")
    @classmethod
    def normalise_address(cls, v: Any) -> str:
        """
        Strip whitespace and lowercase the address before validation.

        We do this in a field validator (not model validator) so that the
        result is available to subsequent validators and the ``domain``
        derivation in the model validator.
        """
        if not isinstance(v, str):
            raise ValueError(f"Email address must be a string, got {type(v)}")
        normalised = v.strip().lower()
        if not _EMAIL_RE.match(normalised):
            raise ValueError(
                f"Invalid email format: {v!r}. "
                "Expected format: local@domain.tld"
            )
        return normalised

    @model_validator(mode="after")
    def derive_domain(self) -> "Email":
        """
        Derive the domain from the address if not explicitly provided.

        Runs after all field validators, so ``self.address`` is guaranteed
        to be the cleaned, lowercase address at this point.
        """
        if self.domain is None and "@" in self.address:
            self.domain = self.address.split("@", 1)[1]
        return self
