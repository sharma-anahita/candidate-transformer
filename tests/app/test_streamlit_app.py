import json

import pytest

import streamlit_app
from src.adapters.github_adapter import GitHubAdapter


def test_parse_json_text_uses_fallback():
    assert streamlit_app.parse_json_text("", {"x": 1}) == {"x": 1}


def test_parse_csv_upload(csv_upload):
    rows = streamlit_app.parse_csv_upload(csv_upload)
    assert rows[0]["first_name"] == "Jane"


def test_parse_json_upload(ats_upload):
    assert streamlit_app.parse_json_upload(ats_upload)["first_name"] == "Jane"


def test_pipeline_helpers_generate_output(normalized_candidate):
    canonical = streamlit_app.merge_sources([normalized_candidate])
    output = streamlit_app.generate_output(canonical, {"fields": {"email": "primary_email"}})
    assert output["email"] == "jane@example.com"


def test_generate_output_validates_schema(normalized_candidate):
    canonical = streamlit_app.merge_sources([normalized_candidate])
    with pytest.raises(ValueError):
        streamlit_app.generate_output(
            canonical,
            {"fields": {"email": "primary_email"}},
            {"type": "object", "required": ["name"]},
        )