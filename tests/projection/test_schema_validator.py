import pytest

from src.projection import JSONSchemaValidator


def test_schema_validator_returns_meaningful_errors():
    payload = {"name": 123}
    schema = {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
        },
    }

    result = JSONSchemaValidator().validate(payload, schema)

    assert result.valid is False
    assert len(result.errors) == 2
    assert any(error.path == "$.name" and "string" in error.message for error in result.errors)
    assert any(error.path == "$" and "'email' is a required property" in error.message for error in result.errors)


def test_schema_validator_can_raise():
    with pytest.raises(ValueError, match="failed JSON Schema validation"):
        JSONSchemaValidator().validate_or_raise(
            {"name": 123},
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )