"""
tests/adapters/test_ats_json_adapter.py — Unit tests for ATSJsonAdapter.

Coverage:
  - get_nested() utility: simple keys, nested keys, array indices, missing keys
  - ATSJsonAdapter construction: empty mapping raises, label stored
  - Happy path: Greenhouse-style nested JSON
  - Happy path: Lever-style flat JSON
  - Happy path: generic flat mapping
  - Email extraction: scalar, array-of-strings, array-of-objects
  - Phone extraction: scalar, array-of-objects, Lever phones list
  - Profile extraction: explicit fields + generic website_url_N
  - Skills: list (Greenhouse tags), delimited string
  - Current role: company + title
  - Location: raw string
  - Missing fields: no crash, no warnings for structurally absent data
  - validate_source: raises AdapterError for non-dict input
  - source_id: built from candidate_id_path
  - Pre-built mappings: GREENHOUSE_FIELD_MAPPING keys resolve correctly
"""

from __future__ import annotations

import pytest

from src.adapters.ats_json_adapter import (
    ATSJsonAdapter,
    GENERIC_FLAT_MAPPING,
    GREENHOUSE_FIELD_MAPPING,
    LEVER_FIELD_MAPPING,
    get_nested,
)
from src.adapters.base import AdapterError
from src.models.profile import Platform
from src.models.provenance import SourceType


# ═════════════════════════════════════════════════════════════════════════════
# get_nested utility
# ═════════════════════════════════════════════════════════════════════════════


class TestGetNested:
    def test_simple_key(self):
        assert get_nested({"a": 1}, "a") == 1

    def test_nested_dict(self):
        assert get_nested({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_array_index(self):
        assert get_nested({"a": [10, 20, 30]}, "a.1") == 20

    def test_nested_array_of_dicts(self):
        data = {"emails": [{"value": "jane@x.com"}, {"value": "jane@y.com"}]}
        assert get_nested(data, "emails.0.value") == "jane@x.com"
        assert get_nested(data, "emails.1.value") == "jane@y.com"

    def test_missing_top_level_key(self):
        assert get_nested({}, "missing") is None

    def test_missing_nested_key(self):
        assert get_nested({"a": {"b": 1}}, "a.c") is None

    def test_out_of_bounds_index(self):
        assert get_nested({"a": [1, 2]}, "a.5") is None

    def test_non_numeric_index_in_list(self):
        assert get_nested({"a": [1, 2]}, "a.x") is None

    def test_path_through_none(self):
        assert get_nested({"a": None}, "a.b") is None

    def test_deep_path(self):
        data = {"x": {"y": {"z": {"w": "found"}}}}
        assert get_nested(data, "x.y.z.w") == "found"

    def test_root_is_none(self):
        assert get_nested(None, "a.b") is None

    def test_integer_value_returned(self):
        assert get_nested({"id": 12345}, "id") == 12345


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — construction
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterConstruction:
    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError):
            ATSJsonAdapter(field_mapping={})

    def test_source_label_stored(self):
        adapter = ATSJsonAdapter(
            field_mapping={"first_name": "first_name"},
            source_label="greenhouse",
        )
        assert adapter._source_label == "greenhouse"

    def test_adapter_name(self):
        adapter = ATSJsonAdapter(field_mapping={"first_name": "first_name"})
        assert adapter.adapter_name == "ATSJsonAdapter"

    def test_source_type(self):
        adapter = ATSJsonAdapter(field_mapping={"first_name": "first_name"})
        assert adapter.source_type == SourceType.ATS_JSON


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — validate_source
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterValidation:
    def test_non_dict_raises_adapter_error(self):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        with pytest.raises(AdapterError):
            adapter.extract("not a dict")  # type: ignore

    def test_list_raises_adapter_error(self):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        with pytest.raises(AdapterError):
            adapter.extract([{"first_name": "Jane"}])  # type: ignore

    def test_empty_dict_accepted(self):
        """Empty dict is valid — all fields will be None."""
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract({})
        assert candidate.first_name is None
        assert candidate.emails == []


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — Greenhouse JSON
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterGreenhouse:
    def test_identity(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert candidate.first_name == "Jane"
        assert candidate.last_name == "Doe"

    def test_emails_from_email_addresses_array(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        emails = [e.address for e in candidate.emails]
        assert "jane.doe@gmail.com" in emails
        assert "jane.doe@acmecorp.com" in emails

    def test_first_email_is_primary(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert candidate.emails[0].is_primary is True

    def test_phone_from_phone_numbers_array(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert len(candidate.phones) >= 1
        assert candidate.phones[0].raw == "+14155551234"

    def test_skills_from_tags(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        skill_names = [s.name for s in candidate.skills]
        assert "Python" in skill_names
        assert "FastAPI" in skill_names

    def test_current_role_from_application(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert len(candidate.experience) >= 1
        exp = candidate.experience[0]
        assert exp.company == "Acme Corp"
        assert exp.title == "Senior Software Engineer"
        assert exp.is_current is True

    def test_profiles_from_website_addresses(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        github = next((p for p in candidate.profiles if p.platform == Platform.GITHUB), None)
        assert github is not None

    def test_source_id_includes_ats_id(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert "greenhouse" in candidate.source_id
        assert "98765" in candidate.source_id

    def test_metadata_contains_source_label(self, greenhouse_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse"
        )
        candidate = adapter.extract(greenhouse_candidate)
        assert candidate.metadata["source_label"] == "greenhouse"


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — Lever JSON
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterLever:
    def test_full_name_from_name_field(self, lever_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=LEVER_FIELD_MAPPING, source_label="lever"
        )
        candidate = adapter.extract(lever_candidate)
        assert candidate.full_name == "John Smith"

    def test_email_from_flat_emails_list(self, lever_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=LEVER_FIELD_MAPPING, source_label="lever"
        )
        candidate = adapter.extract(lever_candidate)
        assert any(e.address == "john.smith@betainc.com" for e in candidate.emails)

    def test_phone_from_lever_phones_list(self, lever_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=LEVER_FIELD_MAPPING, source_label="lever"
        )
        candidate = adapter.extract(lever_candidate)
        assert len(candidate.phones) >= 1
        assert candidate.phones[0].raw == "+12125559876"

    def test_skills_from_tags_list(self, lever_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=LEVER_FIELD_MAPPING, source_label="lever"
        )
        candidate = adapter.extract(lever_candidate)
        skill_names = [s.name for s in candidate.skills]
        assert "JavaScript" in skill_names
        assert "React" in skill_names

    def test_location_from_location_field(self, lever_candidate):
        adapter = ATSJsonAdapter(
            field_mapping=LEVER_FIELD_MAPPING, source_label="lever"
        )
        candidate = adapter.extract(lever_candidate)
        assert candidate.location is not None
        assert "New York" in candidate.location.raw


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — generic flat mapping
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterFlat:
    def test_identity(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        assert candidate.first_name == "Sara"
        assert candidate.last_name == "Connor"

    def test_email(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        assert any(e.address == "sara.connor@example.com" for e in candidate.emails)

    def test_phone(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        assert len(candidate.phones) == 1

    def test_summary(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        assert candidate.summary == "Senior systems engineer."

    def test_skills_from_comma_string(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        skill_names = [s.name for s in candidate.skills]
        assert "Python" in skill_names
        assert "Kubernetes" in skill_names
        assert "Terraform" in skill_names

    def test_explicit_profile_urls(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        platforms = {p.platform for p in candidate.profiles}
        assert Platform.LINKEDIN in platforms
        assert Platform.GITHUB in platforms

    def test_no_warnings_for_valid_data(self, flat_ats_candidate):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract(flat_ats_candidate)
        assert not candidate.has_warnings


# ═════════════════════════════════════════════════════════════════════════════
# ATSJsonAdapter — edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestATSJsonAdapterEdgeCases:
    def test_null_fields_produce_none(self):
        adapter = ATSJsonAdapter(field_mapping=GENERIC_FLAT_MAPPING)
        candidate = adapter.extract({"first_name": None, "email": None})
        assert candidate.first_name is None
        assert candidate.emails == []

    def test_integer_id_converted_to_string_in_source_id(self):
        adapter = ATSJsonAdapter(
            field_mapping=GENERIC_FLAT_MAPPING,
            source_label="ats",
            candidate_id_path="id",
        )
        candidate = adapter.extract({"id": 99999, "first_name": "Test"})
        assert "99999" in candidate.source_id

    def test_string_id_in_source_id(self):
        adapter = ATSJsonAdapter(
            field_mapping=GENERIC_FLAT_MAPPING,
            source_label="lever",
            candidate_id_path="id",
        )
        candidate = adapter.extract({"id": "lever-abc123", "first_name": "Test"})
        assert "lever-abc123" in candidate.source_id

    def test_no_id_field_uses_unknown(self):
        adapter = ATSJsonAdapter(
            field_mapping={"first_name": "first_name"},
            source_label="ats",
        )
        candidate = adapter.extract({"first_name": "Jane"})
        assert "unknown" in candidate.source_id

    def test_duplicate_emails_not_stored_twice(self):
        """Same email from both field_mapping and email_addresses array is deduplicated."""
        adapter = ATSJsonAdapter(
            field_mapping={
                "id": "skip",
                "email_addresses.0.value": "email",
            },
            source_label="gh",
        )
        source = {
            "id": 1,
            "email_addresses": [
                {"value": "jane@x.com", "type": "personal"},
                {"value": "jane@x.com", "type": "personal"},  # duplicate
            ],
        }
        candidate = adapter.extract(source)
        addresses = [e.address for e in candidate.emails]
        assert addresses.count("jane@x.com") == 1

    def test_skills_from_list(self):
        adapter = ATSJsonAdapter(
            field_mapping={"tags": "skills"},
            source_label="ats",
        )
        candidate = adapter.extract({"tags": ["Python", "Docker", "AWS"]})
        assert len(candidate.skills) == 3
        assert candidate.skills[0].name == "Python"

    def test_custom_id_path(self):
        adapter = ATSJsonAdapter(
            field_mapping={"contact.id": "first_name"},
            source_label="ats",
            candidate_id_path="contact.id",
        )
        candidate = adapter.extract({"contact": {"id": "xyz-789"}})
        assert "xyz-789" in candidate.source_id
