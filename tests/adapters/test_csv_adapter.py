"""
tests/adapters/test_csv_adapter.py — Unit tests for CSVAdapter.

Coverage:
  - BaseAdapter contract (abstract method enforcement)
  - Happy path: full row extraction
  - Minimal row: only name + email
  - Custom column mapping
  - Case-insensitive header matching
  - Bad email → warning, not crash
  - Multi-value skills (comma, semicolon, pipe delimited)
  - Profile URL detection (explicit columns + generic website)
  - Structured location columns
  - Raw location string
  - Empty / missing values skipped
  - from_file() reads a real CSV
  - source_type and adapter_name set correctly
  - source_id is deterministic (same row → same ID)
  - validate_source raises AdapterError on wrong type
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from src.adapters.base import AdapterError, BaseAdapter
from src.adapters.csv_adapter import (
    CSVAdapter,
    DEFAULT_COLUMN_MAPPING,
)
from src.utils.parser_utils import split_multivalue
from src.models.extracted_candidate import ExtractedCandidate
from src.models.profile import Platform
from src.models.provenance import SourceType


# ═════════════════════════════════════════════════════════════════════════════
# BaseAdapter contract
# ═════════════════════════════════════════════════════════════════════════════


class TestBaseAdapterContract:
    def test_cannot_instantiate_directly(self):
        """BaseAdapter is abstract — direct instantiation must fail."""
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore

    def test_concrete_subclass_must_implement_source_type_and_extract(self):
        """A subclass missing _extract or source_type cannot be instantiated."""

        class IncompleteAdapter(BaseAdapter):
            pass  # missing source_type and _extract

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_minimal_concrete_adapter(self):
        """A properly implemented subclass can be instantiated."""

        class MinimalAdapter(BaseAdapter[dict]):
            @property
            def source_type(self) -> SourceType:
                return SourceType.CSV

            def _extract(self, source: dict) -> ExtractedCandidate:
                return self._new_candidate("test:source")

        adapter = MinimalAdapter()
        result = adapter.extract({})
        assert isinstance(result, ExtractedCandidate)
        assert result.source_type == SourceType.CSV
        assert result.adapter_name == "MinimalAdapter"


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — construction
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterConstruction:
    def test_default_mapping_loaded(self):
        adapter = CSVAdapter()
        assert adapter._mapping  # non-empty

    def test_custom_mapping_merged_with_defaults(self):
        adapter = CSVAdapter(column_mapping={"Legal First Name": "first_name"})
        # Custom key is present (lowercased)
        assert "legal first name" in adapter._mapping
        # Default keys are still present
        assert "email" in adapter._mapping

    def test_adapter_name(self):
        assert CSVAdapter().adapter_name == "CSVAdapter"

    def test_source_type(self):
        assert CSVAdapter().source_type == SourceType.CSV


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — happy path
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterExtraction:
    def test_full_row_identity(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert candidate.first_name == "Jane"
        assert candidate.last_name == "Doe"

    def test_full_row_email(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert len(candidate.emails) == 1
        assert candidate.emails[0].address == "jane.doe@gmail.com"
        assert candidate.emails[0].is_primary is True

    def test_full_row_phone(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert len(candidate.phones) == 1
        assert candidate.phones[0].raw == "+14155551234"
        assert candidate.phones[0].is_primary is True

    def test_full_row_location(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert candidate.location is not None
        assert candidate.location.raw == "San Francisco, CA"

    def test_full_row_skills(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        skill_names = [s.name for s in candidate.skills]
        assert "Python" in skill_names
        assert "FastAPI" in skill_names
        assert "PostgreSQL" in skill_names

    def test_full_row_summary(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert "7 years" in candidate.summary

    def test_full_row_profiles_github(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        github = next((p for p in candidate.profiles if p.platform == Platform.GITHUB), None)
        assert github is not None
        assert "janedoe" in github.url

    def test_full_row_profiles_linkedin(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        li = next((p for p in candidate.profiles if p.platform == Platform.LINKEDIN), None)
        assert li is not None

    def test_full_row_current_experience(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert len(candidate.experience) == 1
        exp = candidate.experience[0]
        assert exp.company == "Acme Corp"
        assert exp.title == "Senior Software Engineer"
        assert exp.is_current is True

    def test_full_row_no_warnings(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert not candidate.has_warnings

    def test_extraction_id_is_uuid(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        from uuid import UUID
        assert isinstance(candidate.extraction_id, UUID)

    def test_extracted_at_is_utc(self, full_csv_row):
        adapter = CSVAdapter()
        candidate = adapter.extract(full_csv_row)
        assert candidate.extracted_at.tzinfo is not None


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — minimal row
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterMinimalRow:
    def test_minimal_row_identity(self, minimal_csv_row):
        candidate = CSVAdapter().extract(minimal_csv_row)
        assert candidate.first_name == "Alice"
        assert candidate.last_name == "Zhang"

    def test_minimal_row_email(self, minimal_csv_row):
        candidate = CSVAdapter().extract(minimal_csv_row)
        assert len(candidate.emails) == 1

    def test_minimal_row_no_skills(self, minimal_csv_row):
        candidate = CSVAdapter().extract(minimal_csv_row)
        assert candidate.skills == []

    def test_minimal_row_no_experience(self, minimal_csv_row):
        candidate = CSVAdapter().extract(minimal_csv_row)
        assert candidate.experience == []

    def test_minimal_row_no_location(self, minimal_csv_row):
        candidate = CSVAdapter().extract(minimal_csv_row)
        assert candidate.location is None


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — bad data (never-fail contract)
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterBadData:
    def test_invalid_email_produces_warning(self, bad_contact_csv_row):
        candidate = CSVAdapter().extract(bad_contact_csv_row)
        assert candidate.has_warnings
        assert any(w.field == "email" for w in candidate.warnings)

    def test_invalid_email_not_stored(self, bad_contact_csv_row):
        candidate = CSVAdapter().extract(bad_contact_csv_row)
        assert len(candidate.emails) == 0

    def test_valid_phone_still_extracted_despite_bad_email(self, bad_contact_csv_row):
        candidate = CSVAdapter().extract(bad_contact_csv_row)
        assert len(candidate.phones) == 1

    def test_empty_row_raises_adapter_error(self):
        with pytest.raises(AdapterError):
            CSVAdapter().extract({})

    def test_wrong_type_raises_adapter_error(self):
        with pytest.raises(AdapterError):
            CSVAdapter().extract("not a dict")  # type: ignore

    def test_unknown_columns_ignored_gracefully(self):
        row = {
            "unknown_column_xyz": "some value",
            "first_name": "Test",
            "email": "test@example.com",
        }
        candidate = CSVAdapter().extract(row)
        assert candidate.first_name == "Test"


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — custom column mapping
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterCustomMapping:
    def test_custom_mapping_overrides_default(self):
        adapter = CSVAdapter(column_mapping={"Legal First": "first_name"})
        row = {"Legal First": "Jane", "email": "jane@example.com"}
        candidate = adapter.extract(row)
        assert candidate.first_name == "Jane"

    def test_case_insensitive_matching(self):
        row = {"FIRST_NAME": "Jane", "EMAIL": "jane@example.com"}
        candidate = CSVAdapter().extract(row)
        assert candidate.first_name == "Jane"

    def test_whitespace_in_header_tolerated(self):
        row = {"  first_name  ": "Jane", "email": "jane@example.com"}
        candidate = CSVAdapter().extract(row)
        assert candidate.first_name == "Jane"


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — multi-value fields
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterMultiValue:
    def test_comma_separated_skills(self, full_csv_row):
        candidate = CSVAdapter().extract(full_csv_row)
        assert len(candidate.skills) == 3

    def test_semicolon_separated_skills(self, multi_skill_csv_row):
        candidate = CSVAdapter().extract(multi_skill_csv_row)
        skill_names = [s.name for s in candidate.skills]
        assert "Python" in skill_names
        assert "Go" in skill_names
        assert "Rust" in skill_names

    def test_pipe_separated_skills(self, pipe_skill_csv_row):
        candidate = CSVAdapter().extract(pipe_skill_csv_row)
        assert len(candidate.skills) == 3

    def test_multiple_emails_comma_separated(self):
        row = {
            "first_name": "Multi",
            "email": "multi@work.com, multi@home.com",
        }
        candidate = CSVAdapter().extract(row)
        assert len(candidate.emails) == 2
        assert candidate.emails[0].is_primary is True
        assert candidate.emails[1].is_primary is False


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — location handling
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterLocation:
    def test_raw_location_string(self, full_csv_row):
        candidate = CSVAdapter().extract(full_csv_row)
        assert candidate.location.raw == "San Francisco, CA"

    def test_structured_location_columns(self, structured_location_csv_row):
        candidate = CSVAdapter().extract(structured_location_csv_row)
        assert candidate.location.city == "Austin"
        assert candidate.location.state == "Texas"
        assert candidate.location.country == "United States"


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — profile URL auto-detection
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterProfiles:
    def test_explicit_github_url(self, full_csv_row):
        candidate = CSVAdapter().extract(full_csv_row)
        assert any(p.platform == Platform.GITHUB for p in candidate.profiles)

    def test_explicit_linkedin_url(self, full_csv_row):
        candidate = CSVAdapter().extract(full_csv_row)
        assert any(p.platform == Platform.LINKEDIN for p in candidate.profiles)

    def test_generic_website_auto_classified_github(self):
        row = {
            "first_name": "Dev",
            "email": "dev@example.com",
            "website": "https://github.com/devuser",
        }
        candidate = CSVAdapter().extract(row)
        assert any(p.platform == Platform.GITHUB for p in candidate.profiles)

    def test_generic_website_classified_as_other(self):
        row = {
            "first_name": "Dev",
            "email": "dev@example.com",
            "website": "https://myportfolio.io",
        }
        candidate = CSVAdapter().extract(row)
        assert any(p.platform == Platform.OTHER for p in candidate.profiles)

    def test_no_duplicate_profiles(self):
        row = {
            "first_name": "Dev",
            "email": "dev@example.com",
            "github_url": "https://github.com/devuser",
            "website": "https://github.com/devuser",  # same URL
        }
        candidate = CSVAdapter().extract(row)
        github_profiles = [p for p in candidate.profiles if p.platform == Platform.GITHUB]
        assert len(github_profiles) == 1


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — source_id determinism
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterSourceId:
    def test_same_row_same_source_id(self, full_csv_row):
        adapter = CSVAdapter()
        c1 = adapter.extract(full_csv_row)
        c2 = adapter.extract(dict(full_csv_row))
        assert c1.source_id == c2.source_id

    def test_different_rows_different_source_ids(self, full_csv_row, minimal_csv_row):
        adapter = CSVAdapter()
        c1 = adapter.extract(full_csv_row)
        c2 = adapter.extract(minimal_csv_row)
        assert c1.source_id != c2.source_id


# ═════════════════════════════════════════════════════════════════════════════
# CSVAdapter — from_file
# ═════════════════════════════════════════════════════════════════════════════


class TestCSVAdapterFromFile:
    def test_reads_all_valid_rows(self):
        rows = [
            "first_name,last_name,email",
            "Jane,Doe,jane@example.com",
            "John,Smith,john@example.com",
        ]
        content = "\n".join(rows)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        candidates = CSVAdapter.from_file(tmp_path)
        assert len(candidates) == 2
        assert candidates[0].first_name == "Jane"
        assert candidates[1].first_name == "John"

    def test_file_not_found_raises_adapter_error(self):
        with pytest.raises(AdapterError):
            CSVAdapter.from_file("/nonexistent/path/file.csv")

    def test_from_file_uses_sample_csv(self):
        sample_path = Path(__file__).parent.parent.parent / "samples" / "sample_candidates.csv"
        if sample_path.exists():
            candidates = CSVAdapter.from_file(sample_path)
            # At least the valid rows should be extracted
            assert len(candidates) >= 4

    def test_from_file_groups_by_uid(self):
        rows = [
            "uid,fullName,emailAddress,record_type,institutionName,technology,projectName,certificateName",
            "cand_1,Rahul Sharma,rahul@email.com,education,ABC Institute,,,",
            "cand_1,Rahul Sharma,rahul@email.com,skill,,Python,,",
            "cand_1,Rahul Sharma,rahul@email.com,project,,,MyProject,",
            "cand_1,Rahul Sharma,rahul@email.com,certification,,,,Cert1",
            "cand_1,Rahul Sharma,rahul@email.com,certification,,,,Cert1",
        ]
        content = "\n".join(rows)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        candidates = CSVAdapter.from_file(tmp_path)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.full_name == "Rahul Sharma"
        assert candidate.emails[0].address == "rahul@email.com"
        
        assert len(candidate.education) == 1
        assert candidate.education[0].institution == "ABC Institute"
        
        assert len(candidate.skills) == 1
        assert candidate.skills[0].name == "Python"
        
        assert len(candidate.projects) == 1
        assert candidate.projects[0].title == "MyProject"
        
        assert len(candidate.metadata.get("certifications", [])) == 1
        assert candidate.metadata["certifications"][0]["name"] == "Cert1"


# ═════════════════════════════════════════════════════════════════════════════
# _split_multivalue helper
# ═════════════════════════════════════════════════════════════════════════════


class TestSplitMultivalue:
    def test_comma_split(self):
        assert split_multivalue("Python, React, AWS") == ["Python", "React", "AWS"]

    def test_semicolon_split(self):
        assert split_multivalue("Python; React; AWS") == ["Python", "React", "AWS"]

    def test_pipe_split(self):
        assert split_multivalue("Python|React|AWS") == ["Python", "React", "AWS"]

    def test_single_value(self):
        assert split_multivalue("Python") == ["Python"]

    def test_empty_string(self):
        assert split_multivalue("") == []

    def test_strips_whitespace(self):
        assert split_multivalue("  Python  ,  React  ") == ["Python", "React"]

    def test_pipe_takes_priority_over_comma(self):
        # Pipe is checked first, so "Python,Go|Rust" splits on pipe only
        result = split_multivalue("Python,Go|Rust")
        assert result == ["Python,Go", "Rust"]
