"""
adapters/csv_adapter.py — Adapter for structured CSV candidate data.

Responsibility
──────────────
Extract one ExtractedCandidate from one CSV row (a dict[str, str]).
File reading and row iteration happen at the pipeline orchestrator level,
not here — this keeps the adapter focused and independently testable.

Design decisions
────────────────
1. Configurable column mapping
   CSV exports vary wildly across HR tools. Greenhouse calls it "First Name";
   Workday calls it "Legal First Name"; internal tools may call it "fname".
   The column_mapping dict lets the caller declare how their CSV columns map
   to canonical field names. The adapter ships with DEFAULT_COLUMN_MAPPING
   that covers the most common patterns out of the box.

2. Case-insensitive, whitespace-tolerant header matching
   The mapping comparison normalises both the CSV header and the mapping key
   to lowercase with stripped whitespace. This tolerates inconsistent casing
   and accidental spaces in headers.

3. Multi-value fields via delimiter detection
   Skills, emails, and URLs may be stored as comma-separated, semicolon-separated,
   or pipe-separated strings. The adapter auto-detects the delimiter and splits.

4. Profile URL auto-detection
   A generic "website" or "url" column may contain GitHub, LinkedIn, or other
   platform URLs. The adapter inspects the URL and classifies it using the
   ``detect_platform`` utility from the profile model.

5. from_file() convenience classmethod
   Reads an entire CSV file and returns a list of ExtractedCandidates.
   Does NOT change the extraction contract — it just wraps the row-by-row loop.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from src.adapters.base import AdapterError, BaseAdapter
from src.utils.parser_utils import split_multivalue, normalise_key
from src.models.education import Education
from src.models.email import Email, EmailType
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.phone import Phone
from src.models.profile import Platform, Profile, detect_platform
from src.models.provenance import ExtractionMethod, SourceType
from src.models.skill import Skill

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

# Maps CSV column name (user-facing) to canonical field name (internal).
# Keys are matched case-insensitively and with stripped whitespace.
ColumnMapping = dict[str, str]

# ─────────────────────────────────────────────────────────────────────────────
# Default column mapping
# ─────────────────────────────────────────────────────────────────────────────

#: Canonical field names used internally by the adapter.
#: Shared by CSVAdapter and ATSJsonAdapter to allow reuse of _apply_fields().
CANONICAL_FIELDS = frozenset(
    {
        "first_name",
        "last_name",
        "middle_name",
        "full_name",
        "email",
        "phone",
        "location",
        "location_city",
        "location_state",
        "location_country",
        "location_postal_code",
        "summary",
        "skills",
        "current_company",
        "current_title",
        "linkedin_url",
        "github_url",
        "portfolio_url",
        "website_url",
    }
)

#: Ships with the adapter so common CSV exports work without configuration.
#: Keys are lowercased; runtime comparison also lowercases the CSV header.
DEFAULT_COLUMN_MAPPING: ColumnMapping = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "first_name": "first_name",
    "first name": "first_name",
    "firstname": "first_name",
    "given_name": "first_name",
    "given name": "first_name",
    "f_name": "first_name",
    "middle_name": "middle_name",
    "middle name": "middle_name",
    "middlename": "middle_name",
    "last_name": "last_name",
    "last name": "last_name",
    "lastname": "last_name",
    "family_name": "last_name",
    "family name": "last_name",
    "surname": "last_name",
    "l_name": "last_name",
    "full_name": "full_name",
    "fullname": "full_name",
    "name": "full_name",
    "candidate_name": "full_name",
    "candidate name": "full_name",
    "applicant_name": "full_name",
    "applicant name": "full_name",
    # ── Contact ───────────────────────────────────────────────────────────────
    "email": "email",
    "email_address": "email",
    "email address": "email",
    "emailaddress": "email",
    "e-mail": "email",
    "e_mail": "email",
    "mail": "email",
    "phone": "phone",
    "phone_number": "phone",
    "phone number": "phone",
    "mobile": "phone",
    "mobile_number": "phone",
    "mobile number": "phone",
    "cell": "phone",
    "cell_phone": "phone",
    "telephone": "phone",
    "contact_number": "phone",
    "contact number": "phone",
    "contactnumber": "phone",
    # ── Location ──────────────────────────────────────────────────────────────
    "location": "location",
    "currentcity": "location",
    "address": "location",
    "city": "location_city",
    "state": "location_state",
    "province": "location_state",
    "country": "location_country",
    "zip": "location_postal_code",
    "zip_code": "location_postal_code",
    "postal_code": "location_postal_code",
    "postal code": "location_postal_code",
    # ── Professional ──────────────────────────────────────────────────────────
    "summary": "summary",
    "aboutme": "summary",
    "bio": "summary",
    "about": "summary",
    "objective": "summary",
    "profile_summary": "summary",
    "profile summary": "summary",
    "skills": "skills",
    "skill": "skills",
    "technologies": "skills",
    "tech_stack": "skills",
    "tech stack": "skills",
    "current_company": "current_company",
    "current company": "current_company",
    "company": "current_company",
    "employer": "current_company",
    "organization": "current_company",
    "organisation": "current_company",
    "current_title": "current_title",
    "current title": "current_title",
    "title": "current_title",
    "job_title": "current_title",
    "job title": "current_title",
    "position": "current_title",
    "role": "current_title",
    "designation": "current_title",
    # ── Profiles ──────────────────────────────────────────────────────────────
    "linkedin": "linkedin_url",
    "linkedin_url": "linkedin_url",
    "linkedin url": "linkedin_url",
    "linkedin_profile": "linkedin_url",
    "linkedin profile": "linkedin_url",
    "github": "github_url",
    "github_url": "github_url",
    "github url": "github_url",
    "github_profile": "github_url",
    "github profile": "github_url",
    "portfolio": "portfolio_url",
    "portfolio_url": "portfolio_url",
    "portfolio_link": "portfolio_url",
    "website": "website_url",
    "website_url": "website_url",
    "personal_website": "website_url",
    "personal website": "website_url",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _row_source_id(row: dict[str, str]) -> str:
    """
    Generate a deterministic source_id from a CSV row's content.

    Uses SHA-256 of sorted key=value pairs so the same row always produces
    the same ID, enabling deduplication of re-imported CSV files.
    """
    content = "&".join(f"{k}={v}" for k, v in sorted(row.items()))
    return "csv:" + hashlib.sha256(content.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# CSVAdapter
# ─────────────────────────────────────────────────────────────────────────────


class CSVAdapter(BaseAdapter[dict[str, str]]):
    """
    Adapter for structured CSV candidate data.

    Accepts one CSV row as a ``dict[str, str]`` (the format produced by
    ``csv.DictReader``) and returns one ``ExtractedCandidate``.

    Args:
        column_mapping:
            Optional dict mapping CSV column names to canonical field names.
            If None, the DEFAULT_COLUMN_MAPPING is used.
            Custom mappings are MERGED with the default — you only need to
            specify overrides, not a complete replacement.
            Matching is case-insensitive and whitespace-tolerant.

    Example::

        adapter = CSVAdapter(column_mapping={
            "Legal First Name": "first_name",
            "Legal Last Name": "last_name",
            "Work Email": "email",
        })
        row = {"Legal First Name": "Jane", "Legal Last Name": "Doe", ...}
        candidate = adapter.extract(row)
    """

    def __init__(self, column_mapping: Optional[ColumnMapping] = None) -> None:
        # Build effective mapping: defaults overridden by user-provided mapping.
        # Both are stored as lowercased keys for case-insensitive comparison.
        effective: ColumnMapping = {
            normalise_key(k): v for k, v in DEFAULT_COLUMN_MAPPING.items()
        }
        if column_mapping:
            for k, v in column_mapping.items():
                effective[normalise_key(k)] = v
        self._mapping: ColumnMapping = effective

    # ── BaseAdapter interface ─────────────────────────────────────────────────

    @property
    def source_type(self) -> SourceType:
        return SourceType.CSV

    def validate_source(self, source: dict[str, str]) -> None:
        """Ensure source is a non-empty dict."""
        if not isinstance(source, dict):
            raise AdapterError(
                self.adapter_name,
                "unknown",
                f"CSVAdapter requires a dict, got {type(source).__name__}",
            )
        if not source:
            raise AdapterError(
                self.adapter_name,
                "unknown",
                "CSVAdapter received an empty row dict.",
            )

    def _extract(self, source: dict[str, str]) -> ExtractedCandidate:
        """
        Extract candidate data from one CSV row.

        Steps:
          1. Resolve each column in the row against the effective mapping.
          2. Build a canonical_data dict: canonical_field → raw_value.
          3. Populate ExtractedCandidate fields from canonical_data.
          4. Add warnings for any fields that could not be parsed.
        """
        source_id = _row_source_id(source)
        candidate = self._new_candidate(source_id)

        # Step 1: resolve columns → canonical field names
        canonical_data: dict[str, str] = {}
        for col_name, raw_value in source.items():
            normalised_col = normalise_key(col_name)
            canonical_field = self._mapping.get(normalised_col)
            if canonical_field and raw_value and raw_value.strip():
                canonical_data[canonical_field] = raw_value.strip()

        # Step 2: populate candidate fields from resolved data
        self._apply_identity(candidate, canonical_data)
        self._apply_contact(candidate, canonical_data)
        self._apply_location(candidate, canonical_data)
        self._apply_professional(candidate, canonical_data)
        self._apply_profiles(candidate, canonical_data)

        return candidate

    # ── Field group population ────────────────────────────────────────────────

    def _apply_identity(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, str],
    ) -> None:
        """Populate name fields from resolved canonical data."""
        candidate.first_name = data.get("first_name")
        candidate.middle_name = data.get("middle_name")
        candidate.last_name = data.get("last_name")
        candidate.full_name = data.get("full_name")

    def _apply_contact(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, str],
    ) -> None:
        """Populate emails and phones. Handles comma-separated multi-values."""
        # Emails
        raw_email = data.get("email", "")
        if raw_email:
            for addr in split_multivalue(raw_email):
                try:
                    email = Email(
                        address=addr,
                        is_primary=len(candidate.emails) == 0,
                    )
                    candidate.emails.append(email)
                except (ValidationError, ValueError) as exc:
                    candidate.add_warning(
                        field="email",
                        message=f"Could not parse email: {exc}",
                        raw=addr,
                    )

        # Phones
        raw_phone = data.get("phone", "")
        if raw_phone:
            for num in split_multivalue(raw_phone):
                candidate.phones.append(
                    Phone(
                        raw=num,
                        is_primary=len(candidate.phones) == 0,
                    )
                )

    def _apply_location(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, str],
    ) -> None:
        """Populate location from raw string or structured fields."""
        raw = data.get("location")
        city = data.get("location_city")
        state = data.get("location_state")
        country = data.get("location_country")
        postal = data.get("location_postal_code")

        if any([raw, city, state, country, postal]):
            candidate.location = Location(
                raw=raw,
                city=city,
                state=state,
                country=country,
                postal_code=postal,
            )

    def _apply_professional(
        self,
        candidate: ExtractedCandidate,
        data: dict[str, str],
    ) -> None:
        """Populate summary, skills, and current role experience."""
        candidate.summary = data.get("summary")

        # Skills: split multi-value string into Skill objects
        raw_skills = data.get("skills", "")
        if raw_skills:
            for skill_name in split_multivalue(raw_skills):
                if skill_name:
                    candidate.skills.append(
                        Skill(
                            name=skill_name,
                            source_context=f"csv:skills_column:{skill_name}",
                        )
                    )

        # Current role → single Experience entry
        company = data.get("current_company")
        title = data.get("current_title")
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
        data: dict[str, str],
    ) -> None:
        """
        Populate social/professional profile links.

        Handles explicit linkedin_url/github_url columns AND generic
        website_url/portfolio_url columns (auto-detected via detect_platform).
        """
        # Explicit platform URLs
        explicit: list[tuple[str, Platform]] = [
            ("linkedin_url", Platform.LINKEDIN),
            ("github_url", Platform.GITHUB),
            ("portfolio_url", Platform.PORTFOLIO),
        ]
        for field, platform in explicit:
            url = data.get(field, "")
            if url:
                candidate.profiles.append(
                    Profile(platform=platform, url=url)
                )

        # Generic website — classify by domain
        website_url = data.get("website_url", "")
        if website_url:
            platform = detect_platform(website_url)
            # Skip if already captured via an explicit column above
            existing_urls = {p.url for p in candidate.profiles}
            if website_url.rstrip("/") not in existing_urls:
                candidate.profiles.append(
                    Profile(platform=platform, url=website_url)
                )

    def extract_rows(self, rows: list[dict[str, str]]) -> list[ExtractedCandidate]:
        """
        Group rows by candidate UID and aggregate sub-record arrays (education, skills, projects, certifications).
        """
        candidates: list[ExtractedCandidate] = []
        candidates_map: dict[str, ExtractedCandidate] = {}

        for i, row_dict in enumerate(rows):
            # Resolve uid case-insensitively
            uid = ""
            for k, v in row_dict.items():
                if k and k.lower().strip() in ("uid", "uidaddress", "candidate_id", "candidateid"):
                    uid = v.strip()
                    break
            if not uid:
                uid = f"row_{i}"

            if uid not in candidates_map:
                source_id = f"csv:{uid}"
                candidate = self._new_candidate(source_id)
                candidates_map[uid] = candidate
                candidates.append(candidate)

            candidate = candidates_map[uid]

            # Extract basic fields using adapter
            try:
                temp = self.extract(row_dict)
            except Exception as exc:
                logger.warning(
                    "Skipping row %d due to extraction failure: %s",
                    i + 1, exc
                )
                continue

            # Merge basic candidate details
            if not candidate.first_name and temp.first_name:
                candidate.first_name = temp.first_name
            if not candidate.middle_name and temp.middle_name:
                candidate.middle_name = temp.middle_name
            if not candidate.last_name and temp.last_name:
                candidate.last_name = temp.last_name
            if not candidate.full_name and temp.full_name:
                candidate.full_name = temp.full_name

            for email in temp.emails:
                if email.address not in {e.address for e in candidate.emails}:
                    candidate.emails.append(email)

            for phone in temp.phones:
                if phone.raw not in {p.raw for p in candidate.phones}:
                    candidate.phones.append(phone)

            if not candidate.location and temp.location:
                candidate.location = temp.location

            if not candidate.summary and temp.summary:
                candidate.summary = temp.summary

            for warning in temp.warnings:
                candidate.warnings.append(warning)

            # Extract record type specific info
            record_type = ""
            for k, v in row_dict.items():
                if k and k.lower().strip() == "record_type":
                    record_type = v.strip().lower()
                    break

            if record_type == "education":
                institution = ""
                degree = ""
                major = ""
                cgpa = ""
                grading_scale = ""
                start_year = ""
                grad_year = ""

                for k, v in row_dict.items():
                    k_lower = k.lower().strip()
                    if k_lower == "institutionname":
                        institution = v.strip()
                    elif k_lower == "degreename":
                        degree = v.strip()
                    elif k_lower == "major":
                        major = v.strip()
                    elif k_lower == "cgpa":
                        cgpa = v.strip()
                    elif k_lower == "gradingscale":
                        grading_scale = v.strip()
                    elif k_lower == "startyear":
                        start_year = v.strip()
                    elif k_lower == "graduationyear":
                        grad_year = v.strip()

                if institution:
                    gpa = None
                    if cgpa:
                        try:
                            gpa = float(cgpa)
                        except ValueError:
                            pass
                    gpa_scale = 4.0
                    if grading_scale:
                        try:
                            gpa_scale = float(grading_scale)
                        except ValueError:
                            pass
                    edu = Education(
                        institution=institution,
                        degree=degree or None,
                        field_of_study=major or None,
                        gpa=gpa,
                        gpa_scale=gpa_scale,
                        raw_start_date=start_year or None,
                        raw_end_date=grad_year or None
                    )
                    candidate.education.append(edu)

            elif record_type == "skill":
                tech = ""
                for k, v in row_dict.items():
                    k_lower = k.lower().strip()
                    if k_lower in ("technology", "skill"):
                        tech = v.strip()
                        break
                if tech:
                    candidate.skills.append(
                        Skill(
                            name=tech,
                            source_context=f"csv:technology:{tech}",
                        )
                    )

            elif record_type == "project":
                project_name = ""
                stack = ""
                overview = ""
                ongoing = ""

                for k, v in row_dict.items():
                    k_lower = k.lower().strip()
                    if k_lower == "projectname":
                        project_name = v.strip()
                    elif k_lower == "stack":
                        stack = v.strip()
                    elif k_lower == "overview":
                        overview = v.strip()
                    elif k_lower == "ongoing":
                        ongoing = v.strip().lower()

                if project_name:
                    technologies = [t.strip() for t in stack.split(";") if t.strip()] if stack else []
                    is_current = ongoing in ("true", "1", "yes")
                    proj = Experience(
                        company="Personal Project",
                        title=project_name,
                        description=overview or None,
                        is_current=is_current,
                        technologies=technologies
                    )
                    candidate.projects.append(proj)

            elif record_type == "certification":
                cert_name = ""
                issuer = ""

                for k, v in row_dict.items():
                    k_lower = k.lower().strip()
                    if k_lower == "certificatename":
                        cert_name = v.strip()
                    elif k_lower == "issuer":
                        issuer = v.strip()

                if cert_name:
                    if "certifications" not in candidate.metadata:
                        candidate.metadata["certifications"] = []
                    cert = {
                        "name": cert_name,
                        "issuer": issuer or "Unknown"
                    }
                    if cert not in candidate.metadata["certifications"]:
                        candidate.metadata["certifications"].append(cert)

        return candidates

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        column_mapping: Optional[ColumnMapping] = None,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> list[ExtractedCandidate]:
        """
        Read a CSV file and extract all candidates grouped by candidate UID.
        """
        path = Path(file_path)
        adapter = cls(column_mapping=column_mapping)
        rows: list[dict[str, str]] = []

        try:
            with path.open(encoding=encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    row_dict = {k: (v or "") for k, v in row.items() if k}
                    rows.append(row_dict)
        except OSError as exc:
            raise AdapterError(
                adapter.adapter_name,
                str(path),
                f"Cannot open CSV file: {exc}",
            ) from exc

        return adapter.extract_rows(rows)
