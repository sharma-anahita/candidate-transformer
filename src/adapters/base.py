"""
adapters/base.py — Abstract base class for all candidate data adapters.

Architecture
────────────
The Template Method pattern is used here:

  Public API:  extract(source) — called by the pipeline orchestrator
               ↓
  Hook:        validate_source(source) — override to add source validation
               ↓
  Abstract:    _extract(source) — implemented by every concrete adapter

This means subclasses only implement _extract(). The validate_source()
and extract() wrapper are guaranteed to run for all adapters, ensuring
consistent pre/post-processing without code duplication.

Why Generic[S]?
  Each adapter accepts a specific source type:
    CSVAdapter[dict[str, str]]       — one CSV row
    ATSJsonAdapter[dict[str, Any]]   — one ATS candidate JSON object
    ResumeAdapter[Path]              — file path to a PDF/DOCX
    GitHubAdapter[str]               — GitHub profile URL

  Using Generic[S] documents this contract in the type system, enabling
  mypy/pyright to catch mismatches at development time rather than runtime.
  The pipeline orchestrator knows what it feeds each adapter and TypeVar
  ensures that knowledge is type-checked.

Why not raise on missing fields?
  The adapter contract is: never fail. Partial data is valid output.
  An adapter that crashes on a missing field blocks all processing for that
  source, losing all other valid fields. Missing fields = None + warning.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.models.extracted_candidate import ExtractedCandidate
from src.models.provenance import SourceType

logger = logging.getLogger(__name__)

# Type variable for the source type each adapter accepts.
# Bound to Any so adapters can accept any type.
S = TypeVar("S")


class AdapterError(Exception):
    """
    Raised when an adapter encounters a fatal, unrecoverable error.

    Distinct from non-fatal issues (which produce ExtractionWarnings).
    AdapterError is reserved for:
      - Source file not found / unreadable
      - API authentication failure
      - Source format so badly corrupted that nothing can be extracted

    Pipeline orchestrators should catch AdapterError and mark the source
    as failed without crashing the entire pipeline run.
    """

    def __init__(self, adapter_name: str, source_id: str, reason: str) -> None:
        self.adapter_name = adapter_name
        self.source_id = source_id
        self.reason = reason
        super().__init__(
            f"[{adapter_name}] Fatal extraction failure for source '{source_id}': {reason}"
        )


class BaseAdapter(ABC, Generic[S]):
    """
    Abstract base class for all candidate data adapters.

    Contract
    --------
    1. Every subclass returns exactly one ExtractedCandidate from _extract().
    2. _extract() must NEVER raise on missing or malformed field data.
       Use candidate.add_warning() for non-fatal issues.
    3. _extract() must NEVER normalise, merge, or compute confidence.
       It only extracts raw values from the source.
    4. Subclasses must declare source_type so the pipeline can track
       provenance without knowing the adapter's class name.

    Usage
    -----
    Subclasses implement _extract():

        class MyAdapter(BaseAdapter[dict[str, str]]):

            @property
            def source_type(self) -> SourceType:
                return SourceType.CSV

            def _extract(self, source: dict[str, str]) -> ExtractedCandidate:
                candidate = self._new_candidate(source_id="my:source")
                candidate.first_name = source.get("first_name")
                return candidate

    Callers use the public extract() method:

        adapter = MyAdapter()
        candidate = adapter.extract({"first_name": "Jane", ...})
    """

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """
        The SourceType this adapter produces.

        Stored on every ExtractedCandidate so provenance tracking works
        without checking the adapter class at runtime.
        """

    @abstractmethod
    def _extract(self, source: S) -> ExtractedCandidate:
        """
        Extract a single ExtractedCandidate from ``source``.

        This is the only method subclasses MUST implement.
        All extraction logic lives here. No normalisation, no merging.

        Args:
            source: The source data in the format this adapter accepts.

        Returns:
            ExtractedCandidate with all available fields populated.
            Missing fields are None. Non-fatal issues are warnings.

        Raises:
            AdapterError: Only for unrecoverable failures (corrupt source,
                          auth failure). Non-fatal issues use add_warning().
        """

    # ── Template method ───────────────────────────────────────────────────────

    def extract(self, source: S) -> ExtractedCandidate:
        """
        Public extraction entry point — the template method.

        Calls validate_source(), then _extract(). Subclasses should NOT
        override this method. Instead, override _extract() for extraction
        logic and validate_source() for source validation.

        Args:
            source: The source data to extract from.

        Returns:
            ExtractedCandidate produced by _extract().

        Raises:
            AdapterError: Propagated from validate_source() or _extract().
        """
        logger.debug(
            "Starting extraction | adapter=%s source_type=%s",
            self.adapter_name,
            self.source_type.value,
        )
        self.validate_source(source)
        candidate = self._extract(source)
        logger.debug(
            "Extraction complete | adapter=%s warnings=%d",
            self.adapter_name,
            len(candidate.warnings),
        )
        return candidate

    # ── Hooks (override as needed) ────────────────────────────────────────────

    def validate_source(self, source: S) -> None:
        """
        Validate that the source is acceptable before extraction begins.

        Override in subclasses to add source-specific pre-checks.
        Raise AdapterError for unrecoverable problems (e.g., wrong file type).
        Do NOT raise for missing fields — those are handled in _extract().

        Default implementation: no validation (accepts any source).
        """

    # ── Utilities for subclasses ──────────────────────────────────────────────

    @property
    def adapter_name(self) -> str:
        """
        Return the adapter's class name.

        Used as the ``adapter_name`` field on ExtractedCandidate for provenance
        tracking. Returning the class name (not a hardcoded string) means
        renamed adapter classes automatically update provenance without a
        separate change.
        """
        return self.__class__.__name__

    def _new_candidate(self, source_id: str) -> ExtractedCandidate:
        """
        Create a new ExtractedCandidate pre-filled with this adapter's metadata.

        Subclasses call this at the start of _extract() to get a candidate
        with source_type, adapter_name, and source_id already set.

        Args:
            source_id: Unique identifier for the specific source being extracted.

        Returns:
            A new ExtractedCandidate with metadata set and all data fields None/empty.
        """
        return ExtractedCandidate(
            source_type=self.source_type,
            source_id=source_id,
            adapter_name=self.adapter_name,
        )
