from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional
from uuid import uuid5, NAMESPACE_URL

from src.models.canonical_candidate import (
    CanonicalCandidate,
    CanonicalEducation,
    CanonicalEmail,
    CanonicalExperience,
    CanonicalPhone,
    CanonicalProfile,
    CanonicalSkill,
)
from src.models.location import Location
from src.models.normalized_candidate import NormalizedCandidate
from src.models.provenance import ConfidenceField, ExtractionMethod, Provenance
from src.models.skill import Skill


class MergeEngine:
    """
    Merges multiple NormalizedCandidate objects into a partially merged
    CanonicalCandidate.

    This engine deduplicates equivalent facts and preserves unresolved conflicts.
    It intentionally does not decide which conflicting scalar value is correct;
    ConflictResolver owns that step.
    """

    def merge(self, candidates: Iterable[NormalizedCandidate]) -> CanonicalCandidate:
        sources = list(candidates)
        canonical = CanonicalCandidate(
            candidate_id=self._candidate_id(sources),
            merged_from=[c.extraction_id for c in sources],
        )

        self._merge_scalar(canonical, sources, "first_name")
        self._merge_scalar(canonical, sources, "middle_name")
        self._merge_scalar(canonical, sources, "last_name")
        self._merge_summary(canonical, sources)
        self._merge_location(canonical, sources)
        self._merge_emails(canonical, sources)
        self._merge_phones(canonical, sources)
        self._merge_skills(canonical, sources)
        self._merge_experience(canonical, sources)
        self._merge_projects(canonical, sources)
        self._merge_education(canonical, sources)
        self._merge_profiles(canonical, sources)
        return canonical

    def _merge_scalar(
        self,
        canonical: CanonicalCandidate,
        sources: list[NormalizedCandidate],
        field: str,
    ) -> None:
        grouped: dict[str, list[Provenance]] = defaultdict(list)
        display_values: dict[str, str] = {}

        for source in sources:
            value = getattr(source, field)
            if not value:
                continue
            key = self._key(value)
            display_values[key] = value
            grouped[key].append(self._provenance(source, field, value))

        if not grouped:
            return

        first_key = sorted(grouped)[0]
        conflicts = [
            {"value": display_values[key], "provenance": grouped[key]}
            for key in sorted(grouped)
            if key != first_key
        ]
        confidence_field = ConfidenceField[str](
            value=display_values[first_key],
            confidence=0.0,
            provenance=grouped[first_key],
            conflicts=conflicts,
        )
        setattr(canonical, field, confidence_field)
        canonical.provenance[field] = [p for rows in grouped.values() for p in rows]

    def _merge_summary(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped: dict[str, list[Provenance]] = defaultdict(list)
        values: dict[str, str] = {}
        for source in sources:
            if not source.summary:
                continue
            key = self._key(source.summary)
            values[key] = source.summary
            grouped[key].append(self._provenance(source, "summary", source.summary))

        if not grouped:
            return

        first_key = sorted(grouped)[0]
        canonical.summary = ConfidenceField[str](
            value=values[first_key],
            confidence=0.0,
            provenance=grouped[first_key],
            conflicts=[
                {"value": values[key], "provenance": grouped[key]}
                for key in sorted(grouped)
                if key != first_key
            ],
        )
        canonical.provenance["summary"] = [p for rows in grouped.values() for p in rows]

    def _merge_location(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped: dict[str, list[Provenance]] = defaultdict(list)
        values: dict[str, Location] = {}

        for source in sources:
            if not source.location or source.location.is_empty:
                continue
            key = self._location_key(source.location)
            values[key] = source.location
            grouped[key].append(self._provenance(source, "location", source.location.display))

        if not grouped:
            return

        first_key = sorted(grouped)[0]
        canonical.location = ConfidenceField[Location](
            value=values[first_key],
            confidence=0.0,
            provenance=grouped[first_key],
            conflicts=[
                {"value": values[key], "provenance": grouped[key]}
                for key in sorted(grouped)
                if key != first_key
            ],
        )
        canonical.provenance["location"] = [p for rows in grouped.values() for p in rows]

    def _merge_emails(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped: dict[str, tuple[Any, list[Provenance]]] = {}

        for source in sources:
            for email in source.emails:
                key = self._key(email.address)
                provenance = self._provenance(source, "emails", email.address)
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (email, [provenance])

        canonical.emails = [
            CanonicalEmail(**email.model_dump(), confidence=0.0, provenance=provenance)
            for _, (email, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["emails"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_phones(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped: dict[str, tuple[Any, list[Provenance]]] = {}

        for source in sources:
            for phone in source.phones:
                key = phone.normalized or self._digits(phone.raw)
                provenance = self._provenance(source, "phones", phone.normalized or phone.raw)
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (phone, [provenance])

        canonical.phones = [
            CanonicalPhone(**phone.model_dump(), confidence=0.0, provenance=provenance)
            for _, (phone, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["phones"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_skills(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped: dict[str, tuple[Skill, list[Provenance]]] = {}

        for source in sources:
            for skill in source.skills:
                key = self._key(skill.name)
                provenance = self._provenance(
                    source,
                    "skills",
                    skill.name,
                    method=ExtractionMethod.INFERRED if skill.is_inferred else None,
                )
                if key in grouped:
                    grouped[key][1].append(provenance)
                    # Merge github_occurrence_count
                    merged_skill = grouped[key][0]
                    merged_skill.github_occurrence_count = max(
                        getattr(merged_skill, "github_occurrence_count", 1),
                        getattr(skill, "github_occurrence_count", 1)
                    )
                else:
                    grouped[key] = (skill, [provenance])

        canonical.skills = [
            CanonicalSkill(**skill.model_dump(), confidence=0.0, provenance=provenance)
            for _, (skill, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["skills"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_experience(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped = {}

        for source in sources:
            for exp in source.experience:
                key = "|".join([self._key(exp.company), self._key(exp.title), str(exp.start_date or exp.raw_start_date)])
                provenance = self._provenance(source, "experience", f"{exp.title} at {exp.company}")
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (exp, [provenance])

        canonical.experience = [
            CanonicalExperience(**exp.model_dump(), confidence=0.0, provenance=provenance)
            for _, (exp, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["experience"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_projects(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped = {}

        for source in sources:
            for exp in source.projects:
                key = "|".join([self._key(exp.company), self._key(exp.title), str(exp.start_date or exp.raw_start_date)])
                provenance = self._provenance(source, "projects", f"{exp.title} at {exp.company}")
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (exp, [provenance])

        canonical.projects = [
            CanonicalExperience(**exp.model_dump(), confidence=0.0, provenance=provenance)
            for _, (exp, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["projects"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_education(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped = {}

        for source in sources:
            for edu in source.education:
                key = "|".join([self._key(edu.institution), self._key(edu.degree or ""), self._key(edu.field_of_study or "")])
                provenance = self._provenance(source, "education", f"{edu.degree or ''} {edu.institution}".strip())
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (edu, [provenance])

        canonical.education = [
            CanonicalEducation(**edu.model_dump(), confidence=0.0, provenance=provenance)
            for _, (edu, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["education"] = [p for _, rows in grouped.values() for p in rows]

    def _merge_profiles(self, canonical: CanonicalCandidate, sources: list[NormalizedCandidate]) -> None:
        grouped = {}

        for source in sources:
            for profile in source.profiles:
                key = f"{profile.platform}:{profile.username or profile.url}".lower()
                provenance = self._provenance(source, "profiles", profile.url)
                if key in grouped:
                    grouped[key][1].append(provenance)
                else:
                    grouped[key] = (profile, [provenance])

        canonical.profiles = [
            CanonicalProfile(**profile.model_dump(), confidence=0.0, provenance=provenance)
            for _, (profile, provenance) in sorted(grouped.items())
        ]
        canonical.provenance["profiles"] = [p for _, rows in grouped.values() for p in rows]

    def _candidate_id(self, sources: list[NormalizedCandidate]):
        email = next((email.address for c in sources for email in c.emails), None)
        profile = next((profile.url for c in sources for profile in c.profiles), None)
        stable = email or profile or "|".join(sorted(c.source_id for c in sources))
        return uuid5(NAMESPACE_URL, stable)

    def _provenance(
        self,
        source: NormalizedCandidate,
        field: str,
        raw_value: Optional[str],
        method: Optional[ExtractionMethod] = None,
    ) -> Provenance:
        return Provenance(
            source_type=source.source_type,
            adapter_name=source.adapter_name,
            method=method or self._method_for_source(source),
            source_id=source.source_id,
            raw_value=raw_value,
            confidence=0.0,
            extracted_at=source.normalized_at,
            extra={"field": field, "extraction_id": str(source.extraction_id)},
        )

    def _method_for_source(self, source: NormalizedCandidate) -> ExtractionMethod:
        if source.source_type.value in {"csv", "ats_json", "linkedin"}:
            return ExtractionMethod.STRUCTURED_FIELD
        if source.source_type.value == "github":
            return ExtractionMethod.API_RESPONSE
        if source.source_type.value == "manual":
            return ExtractionMethod.MANUAL
        return ExtractionMethod.NLP_HEURISTIC

    def _key(self, value: str) -> str:
        return " ".join(str(value).strip().lower().split())

    def _digits(self, value: str) -> str:
        return "".join(ch for ch in value if ch.isdigit())

    def _location_key(self, location: Location) -> str:
        return self._key(location.display or location.raw or "")