from __future__ import annotations

import copy
import re
from datetime import date, datetime
from typing import Optional

import phonenumbers
from dateutil import parser as date_parser

from src.models.education import DegreeLevel, Education
from src.models.email import Email
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.normalized_candidate import NormalizedCandidate
from src.models.phone import Phone
from src.models.skill import Skill
from src.normalizers.base import BaseNormalizer


class PhoneNormalizer(BaseNormalizer):
    def __init__(self, default_region: str = "US") -> None:
        self.default_region = default_region

    def normalize(self, candidate: NormalizedCandidate) -> None:
        phones = []
        for index, phone in enumerate(candidate.phones):
            try:
                parsed = phonenumbers.parse(phone.raw, self.default_region)
                valid = phonenumbers.is_valid_number(parsed)
                normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164) if valid else None
                phones.append(
                    phone.model_copy(
                        update={
                            "normalized": normalized,
                            "country_code": str(parsed.country_code),
                            "national_number": str(parsed.national_number),
                            "extension": parsed.extension,
                            "is_valid": valid,
                        }
                    )
                )
                candidate.log(f"phones[{index}]", self.name, phone.raw, normalized, valid)
            except Exception as exc:
                phones.append(phone)
                candidate.log(f"phones[{index}]", self.name, phone.raw, None, False, str(exc))
        candidate.phones = phones


class EmailNormalizer(BaseNormalizer):
    def normalize(self, candidate: NormalizedCandidate) -> None:
        emails = []
        for index, email in enumerate(candidate.emails):
            original = email.address
            try:
                normalized = Email(
                    address=email.address,
                    type=email.type,
                    is_primary=email.is_primary,
                    is_verified=email.is_verified,
                )
                emails.append(normalized)
                candidate.log(f"emails[{index}]", self.name, original, normalized.address)
            except Exception as exc:
                candidate.log(f"emails[{index}]", self.name, original, None, False, str(exc))
        candidate.emails = emails


class NameNormalizer(BaseNormalizer):
    def normalize(self, candidate: NormalizedCandidate) -> None:
        original = candidate.display_name
        if candidate.full_name and not candidate.first_name and not candidate.last_name:
            parts = [self._title(p) for p in candidate.full_name.split() if p]
            if len(parts) == 1:
                candidate.first_name = parts[0]
            elif len(parts) == 2:
                candidate.first_name, candidate.last_name = parts
            elif len(parts) > 2:
                candidate.first_name = parts[0]
                candidate.middle_name = " ".join(parts[1:-1])
                candidate.last_name = parts[-1]

        for field in ("first_name", "middle_name", "last_name", "full_name"):
            value = getattr(candidate, field)
            if value:
                setattr(candidate, field, self._title(value))

        candidate.log("name", self.name, original, candidate.display_name)

    def _title(self, value: str) -> str:
        return " ".join(part[:1].upper() + part[1:].lower() for part in value.split())


class CompanyNormalizer(BaseNormalizer):
    SUFFIX_RE = re.compile(r"\b(inc|llc|ltd|corp|corporation|pvt|private|limited)\b\.?", re.I)

    def normalize(self, candidate: NormalizedCandidate) -> None:
        for index, exp in enumerate(candidate.experience):
            original = exp.company
            company = self.SUFFIX_RE.sub("", original).strip(" ,")
            candidate.experience[index] = exp.model_copy(update={"company": company})
            candidate.log(f"experience[{index}].company", self.name, original, company)

        for index, exp in enumerate(candidate.projects):
            original = exp.company
            company = self.SUFFIX_RE.sub("", original).strip(" ,")
            candidate.projects[index] = exp.model_copy(update={"company": company})
            candidate.log(f"projects[{index}].company", self.name, original, company)


class SkillNormalizer(BaseNormalizer):
    ALIASES = {
        "js": "JavaScript",
        "javascript": "JavaScript",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "reactjs": "React",
        "nodejs": "Node.js",
        "postgres": "PostgreSQL",
        "tailwindcss": "Tailwind CSS",
    }

    def normalize(self, candidate: NormalizedCandidate) -> None:
        seen = {}
        for skill in candidate.skills:
            original = skill.name
            canonical = self.ALIASES.get(original.strip().lower(), self._clean(original))
            key = canonical.lower()
            if key not in seen:
                seen[key] = skill.model_copy(update={"name": canonical})
                candidate.log("skills", self.name, original, canonical)
        candidate.skills = list(seen.values())

    def _clean(self, value: str) -> str:
        return value.strip().replace("_", " ").replace("-", " ").title()


class DateNormalizer(BaseNormalizer):
    def normalize(self, candidate: NormalizedCandidate) -> None:
        for index, exp in enumerate(candidate.experience):
            updates = {}
            if exp.raw_start_date:
                updates["start_date"] = self._parse(exp.raw_start_date)
                candidate.log(f"experience[{index}].start_date", self.name, exp.raw_start_date, str(updates["start_date"]))
            if exp.raw_end_date and exp.raw_end_date.lower() not in {"present", "current", "now", "ongoing"}:
                updates["end_date"] = self._parse(exp.raw_end_date)
                candidate.log(f"experience[{index}].end_date", self.name, exp.raw_end_date, str(updates["end_date"]))
            candidate.experience[index] = exp.model_copy(update=updates)

        for index, exp in enumerate(candidate.projects):
            updates = {}
            if exp.raw_start_date:
                updates["start_date"] = self._parse(exp.raw_start_date)
                candidate.log(f"projects[{index}].start_date", self.name, exp.raw_start_date, str(updates["start_date"]))
            if exp.raw_end_date and exp.raw_end_date.lower() not in {"present", "current", "now", "ongoing"}:
                updates["end_date"] = self._parse(exp.raw_end_date)
                candidate.log(f"projects[{index}].end_date", self.name, exp.raw_end_date, str(updates["end_date"]))
            candidate.projects[index] = exp.model_copy(update=updates)

        for index, edu in enumerate(candidate.education):
            updates = {}
            if edu.raw_start_date:
                updates["start_date"] = self._parse(edu.raw_start_date)
            if edu.raw_end_date:
                updates["end_date"] = self._parse(edu.raw_end_date)
            candidate.education[index] = edu.model_copy(update=updates)

    def _parse(self, value: str) -> Optional[date]:
        try:
            return date_parser.parse(value, default=datetime(1900, 1, 1)).date()
        except Exception:
            return None


class EducationNormalizer(BaseNormalizer):
    def normalize(self, candidate: NormalizedCandidate) -> None:
        for index, edu in enumerate(candidate.education):
            level = self._level(edu.degree or "")
            candidate.education[index] = edu.model_copy(update={"degree_level": level})
            candidate.log(f"education[{index}].degree_level", self.name, edu.degree, level.value)

    def _level(self, degree: str) -> DegreeLevel:
        text = degree.lower()
        if any(x in text for x in ("phd", "doctor")):
            return DegreeLevel.DOCTORATE
        if any(x in text for x in ("master", "m.s", "ms ", "mba")):
            return DegreeLevel.MASTER
        if any(x in text for x in ("bachelor", "b.s", "bs ", "btech", "b.tech")):
            return DegreeLevel.BACHELOR
        if "associate" in text:
            return DegreeLevel.ASSOCIATE
        if "certificate" in text:
            return DegreeLevel.CERTIFICATE
        return DegreeLevel.UNKNOWN


class LocationNormalizer(BaseNormalizer):
    def normalize(self, candidate: NormalizedCandidate) -> None:
        if not candidate.location or not candidate.location.raw:
            return
        original = candidate.location.raw
        parts = [p.strip() for p in original.split(",")]
        updates = {}
        if len(parts) >= 1:
            updates["city"] = parts[0]
        if len(parts) >= 2:
            updates["state_code"] = parts[1].upper() if len(parts[1]) <= 3 else None
            updates["state"] = parts[1] if len(parts[1]) > 3 else None
        if len(parts) >= 3:
            updates["country_code"] = parts[2].upper() if len(parts[2]) == 2 else None
            updates["country"] = parts[2] if len(parts[2]) != 2 else None
        candidate.location = candidate.location.model_copy(update=updates)
        candidate.log("location", self.name, original, candidate.location.display)


class CandidateNormalizer:
    def __init__(self, normalizers: list[BaseNormalizer] | None = None) -> None:
        self.normalizers = normalizers or [
            PhoneNormalizer(),
            EmailNormalizer(),
            NameNormalizer(),
            CompanyNormalizer(),
            SkillNormalizer(),
            DateNormalizer(),
            EducationNormalizer(),
            LocationNormalizer(),
        ]

    def normalize(self, extracted: ExtractedCandidate) -> NormalizedCandidate:
        candidate = NormalizedCandidate(
            extraction_id=extracted.extraction_id,
            source_type=extracted.source_type,
            source_id=extracted.source_id,
            adapter_name=extracted.adapter_name,
            first_name=extracted.first_name,
            middle_name=extracted.middle_name,
            last_name=extracted.last_name,
            full_name=extracted.full_name,
            emails=copy.deepcopy(extracted.emails),
            phones=copy.deepcopy(extracted.phones),
            location=copy.deepcopy(extracted.location),
            summary=extracted.summary,
            skills=copy.deepcopy(extracted.skills),
            experience=copy.deepcopy(extracted.experience),
            projects=copy.deepcopy(extracted.projects),
            education=copy.deepcopy(extracted.education),
            profiles=copy.deepcopy(extracted.profiles),
            metadata=copy.deepcopy(extracted.metadata),
        )
        for normalizer in self.normalizers:
            normalizer.normalize(candidate)
        return candidate