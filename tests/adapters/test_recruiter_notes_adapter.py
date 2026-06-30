import pytest

from src.adapters.base import AdapterError
from src.adapters.recruiter_notes_adapter import RecruiterNotesAdapter
from src.models.provenance import SourceType


def test_recruiter_notes_extracts_summary_soft_skills_keywords_and_observations():
    text = """
    Summary: Senior backend engineer, strong ownership and communication.
    Keywords: Python, FastAPI, AWS
    Observations: Impressed with system design; Prefers remote work
    """

    candidate = RecruiterNotesAdapter().extract(text)

    assert candidate.source_type == SourceType.RECRUITER_NOTES
    assert candidate.summary.startswith("Senior backend engineer")
    assert "Ownership" in [s.name for s in candidate.skills]
    assert "Communication" in [s.name for s in candidate.skills]
    assert candidate.metadata["keywords"] == ["Python", "FastAPI", "AWS"]
    assert "Impressed with system design" in candidate.metadata["candidate_observations"]


def test_recruiter_notes_rejects_empty_text():
    with pytest.raises(AdapterError):
        RecruiterNotesAdapter().extract("")