"""
src/utils/parser_utils.py — Shared parsing and sanitization utilities.
"""

from __future__ import annotations

from typing import Any, Optional

_DELIMITERS = ["|", ";", ","]


def split_multivalue(raw: str) -> list[str]:
    """
    Split a multi-value string by auto-detecting the delimiter.

    Checks ``|``, ``;``, then ``,`` in that order of specificity.
    Falls back to treating the entire string as a single value.

    Args:
        raw: Raw multi-value string, e.g., "Python, React | PostgreSQL".

    Returns:
        List of stripped, non-empty strings.
    """
    for delimiter in _DELIMITERS:
        if delimiter in raw:
            return [v.strip() for v in raw.split(delimiter) if v.strip()]
    # Single value
    stripped = raw.strip()
    return [stripped] if stripped else []


def coerce_str(value: Any) -> Optional[str]:
    """
    Coerce any scalar value to string, or None for null/empty values.

    Lists and dicts are not coerced — callers that expect scalars
    should handle lists separately.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    s = str(value).strip()
    return s if s else None


def normalise_key(key: str) -> str:
    """Lowercase and strip a mapping key for comparison."""
    return key.strip().lower()
