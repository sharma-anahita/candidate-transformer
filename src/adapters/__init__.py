"""
src/adapters/__init__.py — Public API of the adapters package.
"""
from src.adapters.github_adapter import GitHubAdapter
from src.adapters.recruiter_notes_adapter import RecruiterNotesAdapter
from src.adapters.ats_json_adapter import (
    ATSJsonAdapter,
    FieldMapping,
    GENERIC_FLAT_MAPPING,
    GREENHOUSE_FIELD_MAPPING,
    LEVER_FIELD_MAPPING,
    get_nested,
)
from src.adapters.base import AdapterError, BaseAdapter
from src.adapters.resume_adapter import ResumeAdapter
from src.adapters.csv_adapter import (
    CANONICAL_FIELDS,
    ColumnMapping,
    CSVAdapter,
    DEFAULT_COLUMN_MAPPING,
)

__all__ = [
    # Base
    "BaseAdapter",
    "AdapterError",
    # CSV
    "CSVAdapter",
    "ColumnMapping",
    "DEFAULT_COLUMN_MAPPING",
    "CANONICAL_FIELDS",
    # Resume
    "ResumeAdapter",
    # ATS JSON
    "ATSJsonAdapter",
    "FieldMapping",
    "GREENHOUSE_FIELD_MAPPING",
    "LEVER_FIELD_MAPPING",
    "GENERIC_FLAT_MAPPING",
    "get_nested",
    "GitHubAdapter",
"RecruiterNotesAdapter",
]
