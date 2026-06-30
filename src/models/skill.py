"""
skill.py — Skill model with category, proficiency, and temporal metadata.

Design rationale:
  Skills are the most consequential field in an ATS. They drive search ranking,
  job matching, and sourcing filters. Yet they arrive in the worst shape:
    - "JS" / "javascript" / "JavaScript" / "ECMAScript" → same thing
    - "Python 3" / "Python" / "Python (Advanced)" → need to be normalised
    - GitHub repos list languages, not skills → must be inferred

  Separating ``name`` (canonical, post-normalisation) from ``aliases`` (alternate
  forms seen in the wild) allows the deduplication engine to collapse variants
  without losing the original evidence.

  ``is_inferred`` distinguishes between skills a candidate explicitly listed
  ("Python — 5 years") and skills inferred from repo contents (found in
  requirements.txt). The confidence engine applies a ceiling to inferred skills
  so they never crowd out explicit claims.

  ``source_context`` stores the raw text fragment that triggered extraction
  (e.g., the sentence "proficient in Django and FastAPI" or the file
  "requirements.txt:django==4.2"). This enables human reviewers to verify
  inferences and the confidence engine to assess signal quality.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class SkillCategory(str, Enum):
    """
    High-level taxonomy for skill classification.

    Used by ATS filter UIs ("show candidates with at least 2 cloud skills"),
    by the confidence engine (programming languages are more verifiable via
    GitHub than soft skills), and by downstream reporting dashboards.

    ``DOMAIN`` captures vertical expertise (e.g., "healthcare", "fintech")
    which is not a technology but is a critical ATS filter for specialist roles.
    """

    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    DATABASE = "database"
    CLOUD = "cloud"
    DEVOPS = "devops"
    DATA_SCIENCE = "data_science"
    DESIGN = "design"
    SOFT_SKILL = "soft_skill"
    DOMAIN = "domain"
    TOOL = "tool"
    PROTOCOL = "protocol"
    METHODOLOGY = "methodology"
    OTHER = "other"
    UNKNOWN = "unknown"


class ProficiencyLevel(str, Enum):
    """
    Self-reported or inferred proficiency level.

    We deliberately avoid numeric scales (1–5, 1–10) because they are
    inconsistently self-calibrated across candidates. A named level with a
    corresponding weight is more transparent and reproducible.

    UNKNOWN is used when the source did not indicate proficiency — e.g.,
    a skills list with no qualification ("Python, React, AWS"). This is
    the most common case and must not be treated as BEGINNER.
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    UNKNOWN = "unknown"


# Maps proficiency level to a normalised weight [0, 1] for confidence scoring.
# These weights are intentionally not configurable at runtime to ensure
# reproducibility. Tuning should be done in code, not in user configs.
PROFICIENCY_WEIGHTS: dict[ProficiencyLevel, float] = {
    ProficiencyLevel.BEGINNER: 0.25,
    ProficiencyLevel.INTERMEDIATE: 0.50,
    ProficiencyLevel.ADVANCED: 0.75,
    ProficiencyLevel.EXPERT: 1.00,
    ProficiencyLevel.UNKNOWN: 0.0,  # 0.0 means weight is not additive
}


# ─────────────────────────────────────────────────────────────────────────────
# Skill model
# ─────────────────────────────────────────────────────────────────────────────


class Skill(BaseModel):
    """
    A single candidate skill with rich metadata.

    Fields
    ------
    name:
        Canonical skill name after normalisation.
        E.g., "JavaScript" not "js", "js", or "java script".
        At extraction time, adapters store the raw name. The skill normaliser
        maps it to the canonical form using the rapidfuzz-backed skill taxonomy.

    aliases:
        Alternate forms of this skill name observed in the source.
        E.g., ["JS", "ECMAScript", "vanilla javascript"].
        Stored so the deduplication engine can collapse duplicates without
        losing evidence. Also useful for debugging normalisation decisions.

    category:
        High-level skill type. Set by the skill normaliser, not the adapter,
        because categorisation requires knowledge of the full taxonomy.
        Defaults to UNKNOWN at extraction time.

    proficiency_level:
        Self-reported or inferred level. Adapters may extract this from
        context clues ("proficient in Python", "basic knowledge of Rust").
        Defaults to UNKNOWN when not stated, which is the common case.

    years_of_experience:
        Duration using this skill across all experience entries, computed
        by the merge engine after temporal analysis of experience data.
        None at extraction time — adapters should not attempt to compute this.

    last_used_year:
        The most recent calendar year this skill was used, derived from
        experience entries. Recency matters: 2-year-old Docker knowledge
        is less valuable than Docker used in the current role.
        None until the merge engine computes it from experience data.

    is_inferred:
        True when this skill was not explicitly stated but derived from
        indirect evidence (repo language stats, package files, README keywords).
        Inferred skills receive a confidence ceiling of 0.70 by default.

    source_context:
        The raw text or file path that evidence this skill.
        E.g., "proficient in Python and Django" (from resume) or
        "requirements.txt:django==4.2,djangorestframework==3.15" (from GitHub).
        Used for audit, human review, and re-ranking.
    """

    name: str
    aliases: list[str] = Field(default_factory=list)
    category: SkillCategory = SkillCategory.UNKNOWN
    proficiency_level: ProficiencyLevel = ProficiencyLevel.UNKNOWN
    years_of_experience: Optional[float] = Field(default=None, ge=0.0)
    last_used_year: Optional[int] = Field(default=None, ge=1970)
    is_inferred: bool = False
    source_context: Optional[str] = None
    github_occurrence_count: int = 1

    model_config = ConfigDict(populate_by_name=True)

    def proficiency_weight(self) -> float:
        """Return the numeric weight for this skill's proficiency level."""
        return PROFICIENCY_WEIGHTS.get(self.proficiency_level, 0.0)
