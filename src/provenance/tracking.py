from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.models.provenance import ConfidenceField, ExtractionMethod, Provenance, SourceType


@dataclass(frozen=True)
class ProvenanceContext:
    source_type: SourceType
    adapter_name: str
    source_id: str
    extraction_method: ExtractionMethod
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0


@dataclass(frozen=True)
class FieldProvenance:
    field: str
    value: Any
    provenance: Provenance


class ProvenanceTracker:
    """
    Reusable provenance builder.

    Every tracked field records:
      - source
      - adapter
      - extraction method
      - timestamp
      - confidence
    """

    def __init__(self, context: ProvenanceContext) -> None:
        self.context = context

    def record(
        self,
        field: str,
        value: Any,
        *,
        raw_value: Optional[str] = None,
        confidence: Optional[float] = None,
        method: Optional[ExtractionMethod] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> FieldProvenance:
        provenance = Provenance(
            source_type=self.context.source_type,
            adapter_name=self.context.adapter_name,
            method=method or self.context.extraction_method,
            source_id=self.context.source_id,
            raw_value=str(raw_value if raw_value is not None else value) if value is not None else None,
            confidence=self._clamp(confidence if confidence is not None else self.context.confidence),
            extracted_at=self.context.timestamp,
            extra={"field": field, **(extra or {})},
        )
        return FieldProvenance(field=field, value=value, provenance=provenance)

    def confidence_field(
        self,
        field: str,
        value: Any,
        *,
        confidence: Optional[float] = None,
        method: Optional[ExtractionMethod] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> ConfidenceField:
        tracked = self.record(field, value, confidence=confidence, method=method, extra=extra)
        return ConfidenceField(
            value=value,
            confidence=tracked.provenance.confidence,
            provenance=[tracked.provenance],
        )

    def batch(self, values: dict[str, Any]) -> dict[str, FieldProvenance]:
        return {field: self.record(field, value) for field, value in values.items() if value is not None}

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))