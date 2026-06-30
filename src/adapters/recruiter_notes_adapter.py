from __future__ import annotations

import hashlib
import re

from src.adapters.base import AdapterError, BaseAdapter
from src.models.extracted_candidate import ExtractedCandidate
from src.models.provenance import SourceType
from src.models.skill import Skill, SkillCategory


SOFT_SKILLS = (
    "communication",
    "collaboration",
    "leadership",
    "ownership",
    "mentoring",
    "empathy",
    "adaptability",
    "curiosity",
    "initiative",
    "problem solving",
    "stakeholder management",
    "teamwork",
)


class RecruiterNotesAdapter(BaseAdapter[str]):
    @property
    def source_type(self) -> SourceType:
        return SourceType.RECRUITER_NOTES

    def validate_source(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise AdapterError(self.adapter_name, "recruiter_notes", "Expected non-empty plain text.")

    def _extract(self, source: str) -> ExtractedCandidate:
        text = source.strip()
        candidate = self._new_candidate(f"recruiter_notes:{_hash(text)}")
        candidate.raw_text = text
        candidate.summary = self._summary(text)

        soft_skills = self._soft_skills(text)
        candidate.skills = [
            Skill(name=s, category=SkillCategory.SOFT_SKILL, source_context="recruiter notes")
            for s in soft_skills
        ]

        candidate.metadata["soft_skills"] = soft_skills
        candidate.metadata["keywords"] = self._keywords(text, soft_skills)
        candidate.metadata["candidate_observations"] = self._observations(text)
        return candidate

    def _summary(self, text: str) -> str:
        labelled = _label(text, "summary")
        if labelled:
            return labelled
        return _sentences(text)[0] if _sentences(text) else text[:280]

    def _soft_skills(self, text: str) -> list[str]:
        lower = text.lower()
        return _dedupe(_title(s) for s in SOFT_SKILLS if s in lower)

    def _keywords(self, text: str, soft_skills: list[str]) -> list[str]:
        labelled = _label(text, "keywords")
        if labelled:
            return _dedupe(x.strip() for x in re.split(r"[,;|]", labelled) if x.strip())

        stop = {"candidate", "interview", "notes", "summary", "strong", "good", "great", "with", "and", "the"}
        soft = {s.lower() for s in soft_skills}
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9+#.-]{1,}\b", text)
        return _dedupe(w.strip(".") for w in words if w.lower() not in stop and w.lower() not in soft)[:20]

    def _observations(self, text: str) -> list[str]:
        labelled = _label(text, "observations") or _label(text, "candidate observations")
        if labelled:
            return _dedupe(x.strip(" -") for x in re.split(r"[;\n]", labelled) if x.strip(" -"))

        cues = ("observed", "noted", "mentioned", "prefers", "concern", "strength", "impressed")
        return [s for s in _sentences(text) if any(cue in s.lower() for cue in cues)]


def _label(text: str, label: str) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\s*:\s*(.+?)(?=^\s*[A-Za-z ]{{2,32}}\s*:|\Z)",
        re.I | re.M | re.S,
    )
    match = pattern.search(text)
    return " ".join(match.group(1).split()) if match else ""


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if x.strip()]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _title(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split())


def _dedupe(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out