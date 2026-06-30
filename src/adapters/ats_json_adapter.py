"""
adapters/ats_json_adapter.py — Adapter for ATS-exported JSON candidate data.

Responsibility
──────────────
Extract one ExtractedCandidate from one ATS candidate JSON object.
Different ATS platforms (Greenhouse, Lever, Ashby, Workday) use completely
different field names for the same information. This adapter is the
configuration-driven mapping layer that bridges that gap.

Design decisions
────────────────
1. Runtime field mapping — zero hardcoded field names
   The adapter never assumes a specific ATS schema. Every field name is
   resolved at runtime from the user-supplied field_mapping config.
   Adding support for a new ATS requires only a new config file, not code.

2. Dot-notation path access (e.g., "contact.email_address")
   ATS JSON payloads are nested. The adapter resolves paths like
   "application.candidate.first_name" without the caller needing to
   flatten the source dict first. Array indices are also supported:
   "email_addresses.0.value" → source["email_addresses"][0]["value"].

3. Multi-value array fields
   Some ATS systems store emails and phones as arrays of objects
   (e.g., Greenhouse: [{"value": "jane@x.com", "type": "work"}]).
   The adapter handles both scalar strings and arrays transparently.

4. Config files ship in configs/
   Greenhouse, Lever, and Ashby mapping configs are included in the
   repository so teams can use them out of the box.

5. source_label for human-readable provenance
   Since the adapter is generic, source_id alone isn't meaningful.
   source_label ("greenhouse", "lever", "ashby") is stored in metadata
   for human-readable provenance in the UI.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import ValidationError

from src.adapters.base import AdapterError, BaseAdapter
from src.utils.parser_utils import coerce_str, split_multivalue
from src.models.education import Education
from src.models.email import Email, EmailType
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.phone import Phone, PhoneType
from src.models.profile import Platform, Profile, detect_platform
from src.models.provenance import ExtractionMethod, SourceType
from src.models.skill import Skill

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

# Maps ATS JSON field path (dot-notation) → canonical field name.
# e.g., {"first_name": "first_name", "contact.email": "email"}
FieldMapping = dict[str, str]

# ─────────────────────────────────────────────────────────────────────────────
# Pre-built field mappings for popular ATS platforms
# ─────────────────────────────────────────────────────────────────────────────

#: Ready-to-use mapping for Greenhouse ATS export format.
GREENHOUSE_FIELD_MAPPING: FieldMapping = {
    "first_name": "first_name",
    "last_name": "last_name",
    "email_addresses.0.value": "email",
    "email_addresses.1.value": "email_secondary",
    "phone_numbers.0.value": "phone",
    "addresses.0.value": "location",
    "website_addresses.0.value": "website_url_0",
    "website_addresses.1.value": "website_url_1",
    "social_media_addresses.0.value": "website_url_2",
    "tags": "skills",
    "application.current_employer": "current_company",
    "application.current_title": "current_title",
    "application.resume_text": "summary",
}

#: Ready-to-use mapping for Lever ATS export format.
LEVER_FIELD_MAPPING: FieldMapping = {
    "name": "full_name",
    "emails.0": "email",
    "phones.0.value": "phone",
    "location": "location",
    "headline": "current_title",
    "summary": "summary",
    "links.0": "website_url_0",
    "links.1": "website_url_1",
    "tags": "skills",
    "origin": "source_label",
}

#: Ready-to-use mapping for a generic flat JSON export (common in internal tools).
GENERIC_FLAT_MAPPING: FieldMapping = {
    "first_name": "first_name",
    "last_name": "last_name",
    "email": "email",
    "phone": "phone",
    "location": "location",
    "summary": "summary",
    "skills": "skills",
    "current_company": "current_company",
    "current_title": "current_title",
    "linkedin_url": "linkedin_url",
    "github_url": "github_url",
    "portfolio_url": "portfolio_url",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_nested(data: Any, path: str) -> Any:
    """
    Access a nested dict/list value using dot-notation path.

    Numeric path segments are treated as array indices.
    Returns None (not raises) for any missing key or out-of-bounds index.

    Examples::

        get_nested({"a": {"b": 1}}, "a.b")           → 1
        get_nested({"a": [1, 2, 3]}, "a.0")           → 1
        get_nested({"a": [{"b": 9}]}, "a.0.b")        → 9
        get_nested({"a": None}, "a.b")                 → None
        get_nested({}, "missing.key")                  → None

    Args:
        data: The root dict or list to traverse.
        path: Dot-separated path string (e.g., "contact.email_addresses.0.value").

    Returns:
        The value at the path, or None if any segment is missing.
    """
    current: Any = data
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            # Scalar hit a non-terminal segment — path is too deep
            return None
    return current



def _resolve_email_type(raw_type: Optional[str]) -> "EmailType":
    """Map ATS email type strings to EmailType enum."""
    from src.models.email import EmailType

    if not raw_type:
        return EmailType.UNKNOWN
    t = raw_type.strip().lower()
    mapping = {
        "work": EmailType.WORK,
        "personal": EmailType.PERSONAL,
        "home": EmailType.PERSONAL,
        "school": EmailType.SCHOOL,
        "other": EmailType.OTHER,
    }
    return mapping.get(t, EmailType.UNKNOWN)


def _resolve_phone_type(raw_type: Optional[str]) -> "PhoneType":
    """Map ATS phone type strings to PhoneType enum."""
    if not raw_type:
        return PhoneType.UNKNOWN
    t = raw_type.strip().lower()
    mapping = {
        "mobile": PhoneType.MOBILE,
        "cell": PhoneType.MOBILE,
        "work": PhoneType.WORK,
        "office": PhoneType.WORK,
        "home": PhoneType.HOME,
        "fax": PhoneType.FAX,
        "other": PhoneType.OTHER,
    }
    return mapping.get(t, PhoneType.UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────────
# ATSJsonAdapter
# ─────────────────────────────────────────────────────────────────────────────


class ATSJsonAdapter(BaseAdapter[dict[str, Any]]):
    """
    Configuration-driven adapter for ATS-exported JSON candidate objects.

    Accepts one candidate JSON object (a ``dict[str, Any]``) and returns
    one ``ExtractedCandidate``. The mapping from ATS field names to canonical
    field names is entirely runtime-driven.

    Args:
        field_mapping:
            Dict mapping ATS JSON paths (dot-notation) to canonical field names.
            Use one of the pre-built mappings (GREENHOUSE_FIELD_MAPPING,
            LEVER_FIELD_MAPPING) or supply your own.

        source_label:
            Human-readable label for the ATS platform (e.g., "greenhouse",
            "lever", "ashby"). Stored in metadata for provenance display.

        candidate_id_path:
            Dot-notation path to the ATS-internal candidate ID in the JSON.
            Used to build a stable source_id. Defaults to "id".

    Example — Greenhouse::

        from src.adapters.ats_json_adapter import ATSJsonAdapter, GREENHOUSE_FIELD_MAPPING

        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING,
            source_label="greenhouse",
            candidate_id_path="id",
        )
        gh_candidate_json = {...}  # from Greenhouse API
        candidate = adapter.extract(gh_candidate_json)
    """

    def __init__(
        self,
        field_mapping: FieldMapping,
        source_label: str = "ats",
        candidate_id_path: str = "id",
    ) -> None:
        if not field_mapping:
            raise ValueError("ATSJsonAdapter requires a non-empty field_mapping.")
        self._field_mapping = field_mapping
        self._source_label = source_label
        self._candidate_id_path = candidate_id_path

    # ── BaseAdapter interface ─────────────────────────────────────────────────

    @property
    def source_type(self) -> SourceType:
        return SourceType.ATS_JSON

    def validate_source(self, source: dict[str, Any]) -> None:
        if not isinstance(source, dict):
            raise AdapterError(
                self.adapter_name,
                "unknown",
                f"ATSJsonAdapter requires a dict, got {type(source).__name__}",
            )

    def _extract(self, source: dict[str, Any]) -> ExtractedCandidate:
        """
        Extract candidate data from an ATS JSON object.

        Steps:
          1. Build source_id from the ATS-internal ID field.
          2. Walk field_mapping and resolve all paths to their values.
          3. Build a canonical_data dict (same shape as CSVAdapter's).
          4. Populate ExtractedCandidate from canonical_data.
          5. Attempt rich extraction for email/phone arrays (ATS-specific).
        """
        # Step 1: build source_id
        ats_id = get_nested(source, self._candidate_id_path)
        source_id = f"{self._source_label}:{ats_id}" if ats_id else f"{self._source_label}:unknown"

        candidate = self._new_candidate(source_id)
        candidate.metadata["source_label"] = self._source_label
        candidate.metadata["ats_id"] = str(ats_id) if ats_id else None

        # Step 2: resolve field_mapping paths
        canonical_data: dict[str, Any] = {}
        for ats_path, canonical_field in self._field_mapping.items():
            value = get_nested(source, ats_path)
            if value is not None and value != "":
                # Accumulate multiple sources for list fields
                if canonical_field in canonical_data:
                    existing = canonical_data[canonical_field]
                    if isinstance(existing, list):
                        existing.append(value)
                    else:
                        canonical_data[canonical_field] = [existing, value]
                else:
                    canonical_data[canonical_field] = value

        # Step 3–4: populate candidate
        self._apply_identity(candidate, canonical_data)
        self._apply_emails(candidate, canonical_data, source)
        self._apply_phones(candidate, canonical_data, source)
        self._apply_location(candidate, canonical_data)
        self._apply_professional(candidate, canonical_data)
        self._apply_profiles(candidate, canonical_data)

        return candidate

    # ── Field population ──────────────────────────────────────────────────────

    def _apply_identity(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
    ) -> None:
        candidate.first_name = coerce_str(data.get("first_name"))
        candidate.middle_name = coerce_str(data.get("middle_name"))
        candidate.last_name = coerce_str(data.get("last_name"))
        candidate.full_name = coerce_str(data.get("full_name"))

    def _apply_emails(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
        raw_source: dict[str, Any],
    ) -> None:
        """
        Populate emails.

        Handles two patterns:
          A) Canonical data has scalar email string(s) from field_mapping.
          B) Source has an email_addresses array of {value, type} objects
             (common in Greenhouse/Lever) — extract all entries directly.
        """
        seen_addresses: set[str] = set()

        def _add_email(addr: str, etype: Optional[str] = None) -> None:
            addr = addr.strip().lower()
            if not addr or addr in seen_addresses:
                return
            seen_addresses.add(addr)
            try:
                candidate.emails.append(
                    Email(
                        address=addr,
                        type=_resolve_email_type(etype),
                        is_primary=len(candidate.emails) == 0,
                    )
                )
            except (ValidationError, ValueError) as exc:
                candidate.add_warning("email", f"Invalid email: {exc}", raw=addr)

        # Pattern A: from field_mapping resolution
        for key in ("email", "email_secondary"):
            val = data.get(key)
            if val:
                if isinstance(val, list):
                    for v in val:
                        _add_email(coerce_str(v) or "")
                else:
                    _add_email(coerce_str(val) or "")

        # Pattern B: native email_addresses array (Greenhouse / Lever style)
        email_array = raw_source.get("email_addresses", [])
        if isinstance(email_array, list):
            for entry in email_array:
                if isinstance(entry, dict):
                    _add_email(
                        entry.get("value", ""),
                        entry.get("type"),
                    )
                elif isinstance(entry, str):
                    _add_email(entry)

    def _apply_phones(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
        raw_source: dict[str, Any],
    ) -> None:
        """
        Populate phones.

        Handles scalar strings from field_mapping AND native phone arrays.
        """
        seen_raws: set[str] = set()

        def _add_phone(raw: str, ptype: Optional[str] = None) -> None:
            raw = raw.strip()
            if not raw or raw in seen_raws:
                return
            seen_raws.add(raw)
            candidate.phones.append(
                Phone(
                    raw=raw,
                    type=_resolve_phone_type(ptype),
                    is_primary=len(candidate.phones) == 0,
                )
            )

        # From field_mapping
        raw_phone = data.get("phone")
        if raw_phone:
            _add_phone(coerce_str(raw_phone) or "")

        # Native phone_numbers array (Greenhouse style)
        phone_array = raw_source.get("phone_numbers", [])
        if isinstance(phone_array, list):
            for entry in phone_array:
                if isinstance(entry, dict):
                    _add_phone(entry.get("value", ""), entry.get("type"))
                elif isinstance(entry, str):
                    _add_phone(entry)

        # Lever style: phones is a list of {value, type} dicts
        lever_phones = raw_source.get("phones", [])
        if isinstance(lever_phones, list):
            for entry in lever_phones:
                if isinstance(entry, dict):
                    _add_phone(entry.get("value", ""), entry.get("type"))

    def _apply_location(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
    ) -> None:
        raw = coerce_str(data.get("location"))
        city = coerce_str(data.get("location_city"))
        state = coerce_str(data.get("location_state"))
        country = coerce_str(data.get("location_country"))

        if any([raw, city, state, country]):
            candidate.location = Location(
                raw=raw,
                city=city,
                state=state,
                country=country,
            )

    def _apply_professional(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
    ) -> None:
        candidate.summary = coerce_str(data.get("summary"))

        # Skills — may be a list (Greenhouse tags) or a delimited string
        raw_skills = data.get("skills")
        if raw_skills:
            skill_names: list[str] = []
            if isinstance(raw_skills, list):
                skill_names = [str(s).strip() for s in raw_skills if s]
            elif isinstance(raw_skills, str):
                # Try comma/semicolon split
                skill_names = split_multivalue(raw_skills)

            for name in skill_names:
                if name:
                    candidate.skills.append(
                        Skill(name=name, source_context=f"ats:{self._source_label}:tag")
                    )

        # Current role
        company = coerce_str(data.get("current_company"))
        title = coerce_str(data.get("current_title"))
        if company or title:
            candidate.experience.append(
                Experience(
                    company=company or "Unknown",
                    title=title or "Unknown",
                    is_current=True,
                )
            )

    def _apply_profiles(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, Any],
    ) -> None:
        """
        Populate profiles from explicit URL fields and generic website_url_N fields.

        Explicit: linkedin_url, github_url, portfolio_url
        Generic: website_url_0, website_url_1, website_url_2 — auto-classified
        """
        seen_urls: set[str] = set()

        def _add_profile(url_raw: Any, platform_hint: Optional[Platform] = None) -> None:
            url = coerce_str(url_raw)
            if not url:
                return
            url = url.rstrip("/")
            if url in seen_urls:
                return
            seen_urls.add(url)
            platform = platform_hint or detect_platform(url)
            candidate.profiles.append(Profile(platform=platform, url=url))

        _add_profile(data.get("linkedin_url"), Platform.LINKEDIN)
        _add_profile(data.get("github_url"), Platform.GITHUB)
        _add_profile(data.get("portfolio_url"), Platform.PORTFOLIO)

        # Auto-classified generic URLs
        for i in range(5):
            _add_profile(data.get(f"website_url_{i}"))
