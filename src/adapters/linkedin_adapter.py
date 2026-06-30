from __future__ import annotations

import hashlib
from typing import Any, Protocol

from src.adapters.base import AdapterError, BaseAdapter
from src.models.education import Education
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.profile import Platform, Profile
from src.models.provenance import SourceType
from src.models.skill import Skill


class LinkedInProfileParser(Protocol):
    def parse(self, source: dict[str, Any]) -> dict[str, Any]:
        ...


class ExportedLinkedInProfileParser:
    """
    Parser for exported LinkedIn profile dictionaries.

    Kept separate from LinkedInAdapter so another parser can replace it later
    without changing adapter or pipeline code.
    """

    def parse(self, source: dict[str, Any]) -> dict[str, Any]:
        profile = source.get("profile", source)

        return {
            "full_name": self._first(profile, "full_name", "name", "formattedName"),
            "headline": self._first(profile, "headline", "occupation", "title"),
            "summary": self._first(profile, "summary", "about", "bio"),
            "profile_url": self._first(profile, "profile_url", "publicProfileUrl", "url"),
            "location": self._first(profile, "location", "geoLocationName"),
            "experience": profile.get("experience") or profile.get("positions") or [],
            "education": profile.get("education") or profile.get("educations") or [],
            "skills": profile.get("skills") or [],
        }

    def _first(self, data: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


class LinkedInAdapter(BaseAdapter[dict[str, Any]]):
    def __init__(self, parser: LinkedInProfileParser | None = None) -> None:
        self.parser = parser or ExportedLinkedInProfileParser()

    @property
    def source_type(self) -> SourceType:
        return SourceType.LINKEDIN

    def validate_source(self, source: dict[str, Any]) -> None:
        if not isinstance(source, dict) or not source:
            raise AdapterError(
                self.adapter_name,
                "linkedin_export",
                "LinkedInAdapter requires a non-empty exported profile dict.",
            )

    def _extract(self, source: dict[str, Any]) -> ExtractedCandidate:
        parsed = self.parser.parse(source)
        source_id = self._source_id(parsed)
        candidate = self._new_candidate(source_id)

        candidate.full_name = parsed.get("full_name")
        candidate.summary = parsed.get("summary")
        candidate.metadata["headline"] = parsed.get("headline")

        profile_url = parsed.get("profile_url")
        if profile_url:
            candidate.profiles.append(
                Profile(
                    platform=Platform.LINKEDIN,
                    url=profile_url,
                    display_name=parsed.get("full_name"),
                    is_verified=False,
                )
            )

        candidate.experience = self._extract_experience(parsed.get("experience") or [])
        candidate.education = self._extract_education(parsed.get("education") or [])
        candidate.skills = self._extract_skills(parsed.get("skills") or [])

        candidate.metadata["linkedin_export"] = {
            "headline": parsed.get("headline"),
            "location": parsed.get("location"),
            "experience_count": len(candidate.experience),
            "education_count": len(candidate.education),
            "skill_count": len(candidate.skills),
        }
        return candidate

    def _extract_experience(self, rows: list[Any]) -> list[Experience]:
        items = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            company = self._first(row, "company", "companyName", "organization")
            title = self._first(row, "title", "role", "position")
            if not company or not title:
                continue
            items.append(
                Experience(
                    company=company,
                    title=title,
                    raw_start_date=self._first(row, "start_date", "startDate", "startedOn"),
                    raw_end_date=self._first(row, "end_date", "endDate", "endedOn"),
                    location=self._first(row, "location"),
                    description=self._first(row, "description", "summary"),
                )
            )
        return items

    def _extract_education(self, rows: list[Any]) -> list[Education]:
        items = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            institution = self._first(row, "institution", "school", "schoolName")
            if not institution:
                continue
            items.append(
                Education(
                    institution=institution,
                    degree=self._first(row, "degree", "degreeName"),
                    field_of_study=self._first(row, "field_of_study", "fieldOfStudy"),
                    raw_start_date=self._first(row, "start_date", "startDate"),
                    raw_end_date=self._first(row, "end_date", "endDate"),
                )
            )
        return items

    def _extract_skills(self, rows: list[Any]) -> list[Skill]:
        skills = []
        for row in rows:
            if isinstance(row, str) and row.strip():
                skills.append(Skill(name=row.strip(), source_context="linkedin skills"))
            elif isinstance(row, dict):
                name = self._first(row, "name", "skillName")
                if name:
                    skills.append(Skill(name=name, source_context="linkedin skills"))
        return skills

    def _first(self, data: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _source_id(self, parsed: dict[str, Any]) -> str:
        stable = parsed.get("profile_url") or parsed.get("full_name") or repr(parsed)
        digest = hashlib.sha256(str(stable).encode("utf-8")).hexdigest()[:16]
        return f"linkedin_export:{digest}"