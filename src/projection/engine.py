from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, Field, model_validator

OMIT_FIELD = object()

class ProjectionError(ValueError):
    """Exception raised when projection fails, containing partial projected data."""
    def __init__(self, message: str, partial_data: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial_data = partial_data

class ProjectionConfig(BaseModel):
    fields: Union[list[dict[str, Any]], dict[str, Any]] = Field(default_factory=list)
    hide_confidence: bool = False
    hide_provenance: bool = False
    include_confidence: bool = True
    include_provenance: bool = True
    on_missing: str = "null"
    missing_value: Any = None
    confidence_range: Any = None
    warnings: list[str] = Field(default_factory=list)

    # Track if metadata inclusion was explicitly defined in config
    include_confidence_defined: bool = False
    include_provenance_defined: bool = False

    @model_validator(mode="before")
    @classmethod
    def track_explicit_definitions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["include_confidence_defined"] = "include_confidence" in data
            data["include_provenance_defined"] = "include_provenance" in data
            
            warnings = []
            if "confidence_range" in data:
                cr = data["confidence_range"]
                is_valid = True
                
                if not isinstance(cr, dict):
                    is_valid = False
                    min_val = None
                    max_val = None
                else:
                    min_val = cr.get("min")
                    max_val = cr.get("max")
                    
                    if min_val is None or max_val is None:
                        is_valid = False
                    elif not (isinstance(min_val, (int, float)) and not isinstance(min_val, bool) and
                              isinstance(max_val, (int, float)) and not isinstance(max_val, bool)):
                        is_valid = False
                    elif not (min_val < max_val):
                        is_valid = False
                
                if not is_valid:
                    warnings.append(
                        f"Invalid confidence range (min={min_val}, max={max_val}). Falling back to default range [0.0, 1.0]."
                    )
                    data["confidence_range"] = {"min": 0.0, "max": 1.0}
            
            data["warnings"] = warnings
        return data

    @classmethod
    def from_runtime(cls, config: dict[str, Any] | str | Path) -> "ProjectionConfig":
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as handle:
                return cls.model_validate(json.load(handle))
        return cls.model_validate(config)


class ProjectionEngine:
    """
    Runtime-configurable JSON projector.

    Supported rule forms (new list-based, and old dict-based).
    """

    def project(self, source: Any, config: ProjectionConfig | dict[str, Any] | str | Path) -> dict[str, Any]:
        cfg = config if isinstance(config, ProjectionConfig) else ProjectionConfig.from_runtime(config)
        
        # Inject confidence/provenance if explicitly defined
        cfg.fields = self._inject_metadata_fields(cfg.fields, cfg)
        
        output = {}
        on_missing = getattr(cfg, "on_missing", "null")
        errors = []

        if isinstance(cfg.fields, list):
            for desc in cfg.fields:
                out_path = desc.get("path")
                src_from = desc.get("from")
                if not out_path and src_from:
                    out_path = src_from
                if not out_path:
                    continue

                try:
                    res = self._project_descriptor(source, desc, cfg, out_path)
                    if res is not OMIT_FIELD:
                        output[out_path] = res
                except ValueError as exc:
                    errors.append(f"Field '{out_path}': {exc}")
        else:
            # Backward compatibility path
            for output_field, rule in cfg.fields.items():
                try:
                    val = self._apply_rule(source, rule, cfg)
                    if val is not None and cfg.confidence_range:
                        if output_field == "confidence" and isinstance(val, (int, float)) and not isinstance(val, bool):
                            c_min = cfg.confidence_range["min"]
                            c_max = cfg.confidence_range["max"]
                            val = round(c_min + val * (c_max - c_min), 4)
                    if val is None:
                        if on_missing == "error":
                            raise ValueError(f"Required field '{output_field}' could not be resolved.")
                        if on_missing == "omit":
                            continue
                    output[output_field] = val
                except ValueError as exc:
                    errors.append(f"Field '{output_field}': {exc}")

        stripped = self._strip_hidden(output, cfg)
        final_json = self._to_json(stripped)

        if errors:
            combined_msg = f"Projection completed with errors: {'; '.join(errors)}"
            raise ProjectionError(combined_msg, final_json)

        return final_json

    def _project_descriptor(self, source: Any, desc: dict[str, Any], config: ProjectionConfig, out_path: str = "") -> Any:
        src_from = desc.get("from")
        required = desc.get("required", False)
        norm_type = desc.get("normalize")
        expected_type = desc.get("type")
        on_missing = getattr(config, "on_missing", "null")

        # Resolve value (fallback to desc.get("path") if "from" is not provided)
        lookup_path = src_from if (src_from is not None and src_from != "") else desc.get("path")
        val = self._missing_safe_get(source, lookup_path, None) if lookup_path else source

        if val is not None and config.confidence_range:
            is_conf = (out_path == "confidence" or 
                       (lookup_path and lookup_path.split(".")[-1] in ("confidence", "overall_confidence")))
            if is_conf and isinstance(val, (int, float)) and not isinstance(val, bool):
                c_min = config.confidence_range["min"]
                c_max = config.confidence_range["max"]
                val = round(c_min + val * (c_max - c_min), 4)

        if val is None:
            if required or on_missing == "error":
                field_label = out_path or lookup_path or "unknown"
                raise ValueError(f"Required field '{field_label}' (from '{lookup_path}') could not be resolved.")
            if on_missing == "omit":
                return OMIT_FIELD
            return None

        # Apply normalization
        if norm_type:
            val = self._apply_normalization(val, norm_type)

        # Handle list-valued properties with fields
        if isinstance(val, list) and "fields" in desc:
            fields = desc["fields"]
            mapped_items = []
            for item in val:
                mapped_item = {}
                if isinstance(fields, list):
                    for nested_desc in fields:
                        nested_path = nested_desc.get("path")
                        res = self._project_descriptor(item, nested_desc, config, nested_path)
                        if res is not OMIT_FIELD:
                            mapped_item[nested_path] = res
                elif isinstance(fields, dict):
                    for out_key, nested_rule in fields.items():
                        mapped_item[out_key] = self._apply_rule(item, nested_rule, config)
                mapped_items.append(mapped_item)
            val = mapped_items
        elif "array" in desc:
            val = self._map_array_descriptor(val, desc["array"], config)
        elif "fields" in desc:
            # Handle nested objects (not list-valued)
            fields = desc["fields"]
            if isinstance(fields, list):
                nested_obj = {}
                for nested_desc in fields:
                    nested_path = nested_desc.get("path")
                    res = self._project_descriptor(val, nested_desc, config, nested_path)
                    if res is not OMIT_FIELD:
                        nested_obj[nested_path] = res
                val = nested_obj
            elif isinstance(fields, dict):
                val = {
                    out_key: self._apply_rule(val, nested_rule, config)
                    for out_key, nested_rule in fields.items()
                }

        # Coerce type
        if expected_type:
            val = self._coerce_type(val, expected_type)

        return val

    def _map_array_descriptor(self, value: Any, array_rule: dict[str, Any], config: ProjectionConfig) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        fields = array_rule.get("fields")
        if not fields:
            return [self._to_json(item) for item in value]

        if isinstance(fields, list):
            mapped_items = []
            for item in value:
                mapped_item = {}
                for desc in fields:
                    out_path = desc.get("path")
                    res = self._project_descriptor(item, desc, config, out_path)
                    if res is not OMIT_FIELD:
                        mapped_item[out_path] = res
                mapped_items.append(mapped_item)
            return mapped_items

        return [
            {
                output_key: self._apply_rule(item, nested_rule, config)
                for output_key, nested_rule in fields.items()
            }
            for item in value
        ]

    def _coerce_type(self, val: Any, expected_type: str) -> Any:
        if val is None:
            return None
        t = expected_type.lower()
        if t in ("string", "str"):
            if isinstance(val, list):
                return "\n".join(str(x) for x in val)
            return str(val)
        if t in ("integer", "int"):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return 0
        if t in ("number", "float"):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        if t in ("boolean", "bool"):
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)
        if t == "array":
            if isinstance(val, list):
                return val
            return [val]
        if t == "object":
            if isinstance(val, dict):
                return val
            return {"value": val}
        return val

    def _apply_normalization(self, val: Any, norm_type: str) -> Any:
        if val is None:
            return None
        n = norm_type.lower()
        if n == "e164":
            import phonenumbers
            try:
                parsed = phonenumbers.parse(str(val), "US")
                if phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            except Exception:
                pass
            return val
        if n == "iso3166":
            s = str(val).strip().lower()
            if len(s) == 2:
                return s.upper()
            common_countries = {
                "india": "IN",
                "united states": "US",
                "united states of america": "US",
                "usa": "US",
                "united kingdom": "GB",
                "uk": "GB",
                "canada": "CA",
                "germany": "DE",
                "france": "FR",
                "australia": "AU",
            }
            return common_countries.get(s, val)
        if n == "canonical":
            s = str(val).strip()
            aliases = {
                "js": "JavaScript",
                "javascript": "JavaScript",
                "ts": "TypeScript",
                "typescript": "TypeScript",
                "reactjs": "React",
                "nodejs": "Node.js",
                "postgres": "PostgreSQL",
                "tailwindcss": "Tailwind CSS",
                "react.js": "React",
                "fastapi": "FastAPI",
                "mongodb": "MongoDB",
            }
            lower_s = s.lower()
            if lower_s in aliases:
                return aliases[lower_s]
            return s.replace("_", " ").replace("-", " ").title()
        return val

    def _apply_rule(self, source: Any, rule: Any, config: ProjectionConfig) -> Any:
        if isinstance(rule, str):
            return self._missing_safe_get(source, rule, config.missing_value)

        if not isinstance(rule, dict):
            return rule

        path = rule.get("path")
        missing = rule.get("missing", config.missing_value)
        value = self._missing_safe_get(source, path, missing) if path else source

        if value is missing:
            return missing

        if "array" in rule:
            return self._map_array(value, rule["array"], config)

        if rule.get("flatten"):
            return self._flatten(self._to_json(value), prefix=rule.get("prefix", ""))

        if "fields" in rule:
            return {
                out_key: self._apply_rule(value, nested_rule, config)
                for out_key, nested_rule in rule["fields"].items()
            }

        return self._to_json(value)

    def _map_array(self, value: Any, array_rule: dict[str, Any], config: ProjectionConfig) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        fields = array_rule.get("fields")
        if not fields:
            return [self._to_json(item) for item in value]

        return [
            {
                output_key: self._apply_rule(item, nested_rule, config)
                for output_key, nested_rule in fields.items()
            }
            for item in value
        ]

    def _missing_safe_get(self, source: Any, path: str | None, missing: Any) -> Any:
        if not path:
            return source

        current = source
        for part in path.split("."):
            if current is None:
                return missing

            if isinstance(current, dict):
                current = current.get(part, missing)
            else:
                if part == "confidence" and hasattr(current, "overall_confidence"):
                    current = getattr(current, "overall_confidence")
                else:
                    current = getattr(current, part, missing)

            if current is missing:
                return missing

        return current

    def _to_json(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        from uuid import UUID
        from datetime import date, datetime

        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()

        if isinstance(value, list):
            return [self._to_json(item) for item in value]

        if isinstance(value, dict):
            return {key: self._to_json(item) for key, item in value.items()}

        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")

        return value

    def _flatten(self, value: Any, prefix: str = "") -> dict[str, Any]:
        value = self._to_json(value)
        if not isinstance(value, dict):
            return {prefix.rstrip("_") or "value": value}

        flat = {}
        for key, item in value.items():
            new_key = f"{prefix}{key}"
            if isinstance(item, dict):
                flat.update(self._flatten(item, prefix=f"{new_key}_"))
            else:
                flat[new_key] = item
        return flat

    def _strip_hidden(self, value: Any, config: ProjectionConfig) -> Any:
        if isinstance(value, list):
            return [self._strip_hidden(item, config) for item in value]

        if isinstance(value, dict):
            hidden = set()
            if config.hide_confidence or not getattr(config, "include_confidence", True):
                hidden.add("confidence")
                hidden.add("field_confidences")
                hidden.add("overall_confidence")
            if config.hide_provenance or not getattr(config, "include_provenance", True):
                hidden.add("provenance")
            return {
                key: self._strip_hidden(item, config)
                for key, item in value.items()
                if key not in hidden
            }

        return value

    def _inject_metadata_fields(self, fields: Any, config: ProjectionConfig) -> Any:
        if isinstance(fields, list):
            has_conf = any(
                isinstance(d, dict) and (d.get("path") == "confidence" or d.get("from") == "confidence")
                for d in fields
            )
            has_prov = any(
                isinstance(d, dict) and (d.get("path") == "provenance" or d.get("from") == "provenance")
                for d in fields
            )

            new_fields = []
            for d in fields:
                if isinstance(d, dict):
                    d_copy = dict(d)
                    if "fields" in d_copy:
                        d_copy["fields"] = self._inject_metadata_fields(d_copy["fields"], config)
                    new_fields.append(d_copy)
                else:
                    new_fields.append(d)

            if config.include_confidence_defined and config.include_confidence and not config.hide_confidence and not has_conf:
                new_fields.append({"path": "confidence", "from": "confidence"})
            if config.include_provenance_defined and config.include_provenance and not config.hide_provenance and not has_prov:
                new_fields.append({"path": "provenance", "from": "provenance"})

            return new_fields

        elif isinstance(fields, dict):
            new_fields = {}
            for k, v in fields.items():
                if isinstance(v, dict):
                    v_copy = dict(v)
                    if "fields" in v_copy:
                        v_copy["fields"] = self._inject_metadata_fields(v_copy["fields"], config)
                    elif "array" in v_copy and isinstance(v_copy["array"], dict) and "fields" in v_copy["array"]:
                        arr = dict(v_copy["array"])
                        arr["fields"] = self._inject_metadata_fields(arr["fields"], config)
                        v_copy["array"] = arr
                    new_fields[k] = v_copy
                else:
                    new_fields[k] = v

            if config.include_confidence_defined and config.include_confidence and not config.hide_confidence and "confidence" not in new_fields:
                new_fields["confidence"] = "confidence"
            if config.include_provenance_defined and config.include_provenance and not config.hide_provenance and "provenance" not in new_fields:
                new_fields["provenance"] = "provenance"

            return new_fields

        return fields