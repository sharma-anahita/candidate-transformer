from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.canonical_candidate import CanonicalCandidate
from src.models.provenance import ConfidenceField, Provenance, SourceType


@dataclass(frozen=True)
class ConflictResolutionRule:
    name: str
    explanation: str


class ConflictResolver:
    """
    Resolves conflicts already preserved by MergeEngine.

    Deterministic rules, in order:
      1. Structured source wins over unstructured source.
      2. Higher confidence wins.
      3. Most recent source wins.
      4. Longest normalized company name wins for company-bearing records.
      5. Lexical order breaks any remaining tie.

    No randomness, no current clock, no model calls.
    """

    rules = [
        ConflictResolutionRule(
            "Structured > Unstructured",
            "ATS, CSV, LinkedIn, GitHub, and manual sources are favored over resume and recruiter-note text because their field boundaries are explicit.",
        ),
        ConflictResolutionRule(
            "Higher confidence wins",
            "When the ConfidenceEngine has scored a value, the highest confidence value wins.",
        ),
        ConflictResolutionRule(
            "Most recent wins",
            "If reliability and confidence tie, the value from the newest provenance timestamp wins.",
        ),
        ConflictResolutionRule(
            "Longest normalized company name wins",
            "For experience conflicts, a longer normalized company name usually preserves the most specific legal or brand name.",
        ),
        ConflictResolutionRule(
            "Lexical tie-break",
            "If all meaningful signals tie, sort by value text and choose the first so results are stable across runs.",
        ),
    ]

    STRUCTURED_SOURCES = {
        SourceType.ATS_JSON,
        SourceType.CSV,
        SourceType.LINKEDIN,
        SourceType.GITHUB,
        SourceType.MANUAL,
    }

    def resolve(self, candidate: CanonicalCandidate) -> CanonicalCandidate:
        data = candidate.model_copy(deep=True)

        for field in ("first_name", "middle_name", "last_name", "summary", "location"):
            confidence_field = getattr(data, field)
            if confidence_field is not None:
                setattr(data, field, self._resolve_confidence_field(confidence_field))

        data.experience = sorted(
            data.experience,
            key=lambda exp: (
                exp.start_date is None,
                exp.start_date or exp.raw_start_date or "",
                exp.company.lower(),
                exp.title.lower(),
            ),
            reverse=True,
        )
        return data

    def _resolve_confidence_field(self, field: ConfidenceField) -> ConfidenceField:
        if not field.conflicts:
            return field

        options = [{"value": field.value, "provenance": field.provenance}]
        options.extend(field.conflicts)
        winner = sorted(options, key=self._option_rank, reverse=True)[0]

        losing = [
            option
            for option in options
            if self._value_text(option["value"]) != self._value_text(winner["value"])
        ]

        return ConfidenceField(
            value=winner["value"],
            confidence=self._provenance_confidence(winner["provenance"]),
            provenance=winner["provenance"],
            conflicts=losing,
        )

    def _option_rank(self, option: dict[str, Any]) -> tuple:
        provenance = option.get("provenance") or []
        value = option.get("value")
        return (
            self._structured_score(provenance),
            self._provenance_confidence(provenance),
            self._most_recent_timestamp(provenance),
            self._company_name_length(value),
            self._value_text(value),
        )

    def _structured_score(self, provenance: list[Provenance]) -> int:
        return max((1 if p.source_type in self.STRUCTURED_SOURCES else 0 for p in provenance), default=0)

    def _provenance_confidence(self, provenance: list[Provenance]) -> float:
        return max((p.confidence for p in provenance), default=0.0)

    def _most_recent_timestamp(self, provenance: list[Provenance]) -> float:
        return max((p.extracted_at.timestamp() for p in provenance), default=0.0)

    def _company_name_length(self, value: Any) -> int:
        company = getattr(value, "company", None)
        return len(" ".join(company.split())) if company else 0

    def _value_text(self, value: Any) -> str:
        if hasattr(value, "display"):
            return value.display.lower()
        return str(value).lower()