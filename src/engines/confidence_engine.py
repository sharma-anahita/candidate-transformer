from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from src.models.canonical_candidate import CanonicalCandidate
from src.models.provenance import ExtractionMethod, Provenance, SourceType


@dataclass(frozen=True)
class ConfidenceFormula:
    """
    Confidence formula:

    source_confidence = source_weight * method_weight * quality_multiplier

    field_confidence =
        clamp(max(source_confidence votes) + agreement_bonus - conflict_penalty)

    overall_confidence =
        weighted_average(field_confidence[field] * field_importance[field])

    Cross-source agreement increases confidence because independent sources
    corroborate the same fact. Conflicts reduce confidence because unresolved
    disagreement makes the canonical value less certain.
    """

    source_weights: dict[SourceType, float] = field(
        default_factory=lambda: {
            SourceType.ATS_JSON: 0.95,
            SourceType.CSV: 0.90,
            SourceType.LINKEDIN: 0.88,
            SourceType.GITHUB: 0.82,
            SourceType.MANUAL: 0.80,
            SourceType.RESUME: 0.80,
            SourceType.RECRUITER_NOTES: 0.62,
        }
    )
    method_weights: dict[ExtractionMethod, float] = field(
        default_factory=lambda: {
            ExtractionMethod.API_RESPONSE: 0.98,
            ExtractionMethod.STRUCTURED_FIELD: 0.95,
            ExtractionMethod.MANUAL: 0.90,
            ExtractionMethod.REGEX: 0.78,
            ExtractionMethod.NLP_HEURISTIC: 0.70,
            ExtractionMethod.INFERRED: 0.62,
        }
    )
    field_importance: dict[str, float] = field(
        default_factory=lambda: {
            "emails": 1.00,
            "phones": 0.85,
            "first_name": 0.75,
            "last_name": 0.75,
            "location": 0.55,
            "summary": 0.40,
            "skills": 0.80,
            "experience": 0.90,
            "projects": 0.70,
            "education": 0.60,
            "profiles": 0.45,
        }
    )
    agreement_bonus_per_extra_source: float = 0.04
    max_agreement_bonus: float = 0.12
    conflict_penalty_per_conflict: float = 0.08
    inferred_confidence_ceiling: float = 0.70


class ConfidenceEngine:
    def __init__(self, formula: ConfidenceFormula | None = None) -> None:
        self.formula = formula or ConfidenceFormula()

    def score(self, candidate: CanonicalCandidate) -> CanonicalCandidate:
        scored = candidate.model_copy(deep=True)

        self._score_scalar(scored, "first_name")
        self._score_scalar(scored, "middle_name")
        self._score_scalar(scored, "last_name")
        self._score_scalar(scored, "summary")
        self._score_scalar(scored, "location")

        scored.emails = [item.model_copy(update={"confidence": self._item_confidence(item.provenance)}) for item in scored.emails]
        scored.phones = [item.model_copy(update={"confidence": self._item_confidence(item.provenance, valid=getattr(item, "is_valid", True))}) for item in scored.phones]
        scored.skills = [
            item.model_copy(
                update={
                    "confidence": self._item_confidence(
                        item.provenance,
                        inferred=item.is_inferred,
                        is_skill=True,
                        github_occurrence_count=item.github_occurrence_count,
                    )
                }
            )
            for item in scored.skills
        ]
        scored.experience = [item.model_copy(update={"confidence": self._item_confidence(item.provenance)}) for item in scored.experience]
        scored.projects = [item.model_copy(update={"confidence": self._item_confidence(item.provenance)}) for item in scored.projects]
        scored.education = [item.model_copy(update={"confidence": self._item_confidence(item.provenance)}) for item in scored.education]
        scored.profiles = [item.model_copy(update={"confidence": self._item_confidence(item.provenance)}) for item in scored.profiles]

        scored.field_confidences = self._field_confidences(scored)
        scored.overall_confidence = self._overall_confidence(scored.field_confidences)
        self._refresh_flat_provenance(scored)
        return scored

    def source_confidence(self, provenance: Provenance, quality_multiplier: float = 1.0, is_skill: bool = False) -> float:
        source_weight = self.formula.source_weights.get(provenance.source_type, 0.50)
        if is_skill and provenance.source_type == SourceType.GITHUB:
            source_weight = 0.95
        method_weight = self.formula.method_weights.get(provenance.method, 0.50)
        return self._clamp(source_weight * method_weight * quality_multiplier)

    def _score_scalar(self, candidate: CanonicalCandidate, field: str) -> None:
        value = getattr(candidate, field)
        if value is None:
            return

        provenance = value.provenance
        if field in ("first_name", "middle_name", "last_name"):
            provenance = [p for p in provenance if p.source_type != SourceType.GITHUB]

        confidence = self._field_confidence(
            provenance,
            conflict_count=len(value.conflicts),
            inferred=value.is_inferred,
        )
        setattr(candidate, field, value.model_copy(update={"confidence": confidence}))

    def _item_confidence(
        self,
        provenance: list[Provenance],
        inferred: bool = False,
        valid: bool = True,
        is_skill: bool = False,
        github_occurrence_count: int = 1,
    ) -> float:
        quality = 1.0 if valid else 0.65
        return self._field_confidence(
            provenance,
            quality_multiplier=quality,
            inferred=inferred,
            is_skill=is_skill,
            github_occurrence_count=github_occurrence_count,
        )

    def _field_confidence(
        self,
        provenance: list[Provenance],
        conflict_count: int = 0,
        quality_multiplier: float = 1.0,
        inferred: bool = False,
        is_skill: bool = False,
        github_occurrence_count: int = 1,
    ) -> float:
        if not provenance:
            return 0.0

        source_votes = [self.source_confidence(p, quality_multiplier, is_skill=is_skill) for p in provenance]
        base = max(source_votes)
        distinct_sources = len({(p.source_type, p.source_id) for p in provenance})
        agreement_bonus = min(
            max(0, distinct_sources - 1) * self.formula.agreement_bonus_per_extra_source,
            self.formula.max_agreement_bonus,
        )
        conflict_penalty = conflict_count * self.formula.conflict_penalty_per_conflict
        confidence = self._clamp(base + agreement_bonus - conflict_penalty)

        has_github = any(p.source_type == SourceType.GITHUB for p in provenance)
        has_non_github = any(p.source_type != SourceType.GITHUB for p in provenance)
        is_matched_github_skill = has_github and has_non_github

        if inferred and not is_matched_github_skill:
            confidence = min(confidence, self.formula.inferred_confidence_ceiling)

        if is_matched_github_skill:
            confidence = self._clamp(confidence + 0.10)

        if is_skill and has_github and github_occurrence_count > 1:
            confidence = self._clamp(confidence + 0.05)

        return confidence

    def _field_confidences(self, candidate: CanonicalCandidate) -> dict[str, float]:
        values = {}

        for field in ("first_name", "middle_name", "last_name", "summary", "location"):
            confidence_field = getattr(candidate, field)
            if confidence_field is not None:
                values[field] = confidence_field.confidence

        if candidate.emails:
            values["emails"] = max(email.confidence for email in candidate.emails)
        if candidate.phones:
            values["phones"] = max(phone.confidence for phone in candidate.phones)
        if candidate.skills:
            values["skills"] = mean(skill.confidence for skill in candidate.skills)
        if candidate.experience:
            values["experience"] = mean(exp.confidence for exp in candidate.experience)
        if candidate.projects:
            values["projects"] = mean(proj.confidence for proj in candidate.projects)
        if candidate.education:
            values["education"] = mean(edu.confidence for edu in candidate.education)
        if candidate.profiles:
            values["profiles"] = mean(profile.confidence for profile in candidate.profiles)

        return values

    def _overall_confidence(self, field_confidences: dict[str, float]) -> float:
        numerator = 0.0
        denominator = 0.0

        for field, confidence in field_confidences.items():
            weight = self.formula.field_importance.get(field, 0.25)
            numerator += confidence * weight
            denominator += weight

        if denominator == 0:
            return 0.0
        return self._clamp(numerator / denominator)

    def _refresh_flat_provenance(self, candidate: CanonicalCandidate) -> None:
        provenance = {}

        for field in ("first_name", "middle_name", "last_name", "summary", "location"):
            confidence_field = getattr(candidate, field)
            if confidence_field is not None:
                provenance[field] = confidence_field.provenance

        provenance["emails"] = [p for item in candidate.emails for p in item.provenance]
        provenance["phones"] = [p for item in candidate.phones for p in item.provenance]
        provenance["skills"] = [p for item in candidate.skills for p in item.provenance]
        provenance["experience"] = [p for item in candidate.experience for p in item.provenance]
        provenance["projects"] = [p for item in candidate.projects for p in item.provenance]
        provenance["education"] = [p for item in candidate.education for p in item.provenance]
        provenance["profiles"] = [p for item in candidate.profiles for p in item.provenance]
        candidate.provenance = {key: value for key, value in provenance.items() if value}

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, round(value, 4)))