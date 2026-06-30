"""
phone.py — Phone number model.

Design rationale:
  Phone numbers are extracted in wildly inconsistent formats:
    "(415) 555-1234"  "+1-415-555-1234"  "4155551234"  "+44 20 7946 0958"

  We separate ``raw`` (what the adapter found) from ``normalized`` (E.164 format,
  computed by the phone normaliser using the ``phonenumbers`` library).

  Why store raw alongside normalised?
    1. Normalisation can fail or produce wrong output (e.g., ambiguous country code).
       The raw value lets the normaliser re-attempt with better context hints.
    2. Provenance.raw_value stores it at the field level, but Phone.raw makes it
       directly accessible without traversing the provenance chain — useful in the
       merge engine for deduplication (compare normalised first, fallback to raw).

  E.164 is chosen as the canonical format because it is:
    - Unambiguous globally (always includes country code)
    - Machine-parseable without locale context
    - The standard used by Twilio, AWS SNS, and most telephony APIs
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class PhoneType(str, Enum):
    """
    Classifies the role of this phone number.

    Mobile numbers are preferred for outreach (SMS, WhatsApp).
    Work numbers may be switchboards — not ideal for direct candidate contact.
    Home/fax are rarely provided but must be stored when present.
    """

    MOBILE = "mobile"
    WORK = "work"
    HOME = "home"
    FAX = "fax"
    OTHER = "other"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Phone model
# ─────────────────────────────────────────────────────────────────────────────


class Phone(BaseModel):
    """
    A single phone number with both raw and normalised representations.

    Fields
    ------
    raw:
        The original phone number string exactly as extracted from the source.
        Never modified. Stored so the normaliser can retry parsing with country
        hints if the initial parse failed or produced an invalid number.

    normalized:
        E.164 format: ``+[country_code][subscriber_number]``.
        Example: ``+14155551234``.
        Populated by the phone normaliser. ``None`` if parsing failed.
        This is the field used for deduplication and outreach — never ``raw``.

    country_code:
        The ITU-T E.164 country calling code as a string (not int), because
        leading zeros are significant in some territories and int would drop them.
        E.g., ``"1"`` (US/CA), ``"44"`` (UK), ``"91"`` (India).
        Used for carrier/region routing and jurisdiction-aware compliance.

    national_number:
        The subscriber number without the country code.
        Used for display formatting: ``+14155551234`` → ``(415) 555-1234`` (US format).
        Formatting is locale-specific — the normaliser handles this using the
        ``phonenumbers`` library's ``format_number`` function.

    extension:
        PBX extension suffix, e.g., ``"ext. 204"``.
        Extracted from strings like ``"+1 (415) 555-1234 x204"``.
        Stored separately so the core number can be normalised independently.

    type:
        Mobile / work / home / fax.  Often inferrable from context in the source
        text ("Cell: ...", "Office: ...").

    is_primary:
        Flags the single phone to use for outreach.
        Set by the merge engine, not by adapters.

    is_valid:
        Set by the phonenumbers library after parsing.
        A number can be parseable but still invalid (e.g., wrong number of digits).
        Invalid numbers are stored (they came from the source) but confidence
        scoring penalises fields with ``is_valid=False``.
    """

    raw: str
    normalized: Optional[str] = None
    country_code: Optional[str] = None
    national_number: Optional[str] = None
    extension: Optional[str] = None
    type: PhoneType = PhoneType.UNKNOWN
    is_primary: bool = False
    is_valid: bool = False

    model_config = ConfigDict(populate_by_name=True)
