from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class SchemaValidationError:
    path: str
    schema_path: str
    message: str
    validator: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[SchemaValidationError] = field(default_factory=list)


class JSONSchemaValidator:
    def validate(self, payload: dict[str, Any], schema: dict[str, Any]) -> ValidationResult:
        validator = Draft202012Validator(schema)
        errors = []

        for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
            errors.append(
                SchemaValidationError(
                    path=self._format_path(error.path),
                    schema_path=self._format_path(error.schema_path),
                    message=error.message,
                    validator=error.validator,
                )
            )

        return ValidationResult(valid=not errors, errors=errors)

    def validate_or_raise(self, payload: dict[str, Any], schema: dict[str, Any]) -> None:
        result = self.validate(payload, schema)
        if not result.valid:
            joined = "; ".join(f"{error.path}: {error.message}" for error in result.errors)
            raise ValueError(f"Projected output failed JSON Schema validation: {joined}")

    def _format_path(self, path) -> str:
        parts = list(path)
        if not parts:
            return "$"
        return "$." + ".".join(str(part) for part in parts)