from src.models.provenance import ExtractionMethod, SourceType
from src.provenance import ProvenanceContext, ProvenanceTracker


def test_provenance_tracker_records_every_required_field():
    tracker = ProvenanceTracker(
        ProvenanceContext(
            source_type=SourceType.LINKEDIN,
            adapter_name="LinkedInAdapter",
            source_id="linkedin:1",
            extraction_method=ExtractionMethod.STRUCTURED_FIELD,
            confidence=0.88,
        )
    )

    tracked = tracker.record("headline", "Senior Engineer")

    assert tracked.provenance.source_type == SourceType.LINKEDIN
    assert tracked.provenance.adapter_name == "LinkedInAdapter"
    assert tracked.provenance.method == ExtractionMethod.STRUCTURED_FIELD
    assert tracked.provenance.raw_value == "Senior Engineer"
    assert tracked.provenance.confidence == 0.88
    assert tracked.provenance.extracted_at is not None