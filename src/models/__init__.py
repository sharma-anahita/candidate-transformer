"""
src/models/__init__.py — Public API of the models package.

Importing from ``src.models`` gives access to all canonical models
without needing to know which submodule they live in.
"""

from src.models.canonical_candidate import (
    CanonicalCandidate,
    CanonicalEducation,
    CanonicalEmail,
    CanonicalExperience,
    CanonicalPhone,
    CanonicalProfile,
    CanonicalSkill,
)
from src.models.education import DegreeLevel, Education
from src.models.email import Email, EmailType
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate, ExtractionWarning
from src.models.location import Location
from src.models.normalized_candidate import NormalizedCandidate, NormalizationLog
from src.models.phone import Phone, PhoneType
from src.models.profile import DOMAIN_TO_PLATFORM, Platform, Profile, detect_platform
from src.models.provenance import (
    ConfidenceField,
    ExtractionMethod,
    Provenance,
    SourceType,
)
from src.models.skill import (
    PROFICIENCY_WEIGHTS,
    ProficiencyLevel,
    Skill,
    SkillCategory,
)

__all__ = [
    # Provenance
    "SourceType",
    "ExtractionMethod",
    "Provenance",
    "ConfidenceField",
    # Contact
    "EmailType",
    "Email",
    "PhoneType",
    "Phone",
    # Location
    "Location",
    # Profile
    "Platform",
    "Profile",
    "DOMAIN_TO_PLATFORM",
    "detect_platform",
    # Skill
    "SkillCategory",
    "ProficiencyLevel",
    "PROFICIENCY_WEIGHTS",
    "Skill",
    # Experience
    "Experience",
    # Education
    "DegreeLevel",
    "Education",
    # Pipeline stages
    "ExtractionWarning",
    "ExtractedCandidate",
    "NormalizationLog",
    "NormalizedCandidate",
    # Canonical
    "CanonicalEmail",
    "CanonicalPhone",
    "CanonicalSkill",
    "CanonicalExperience",
    "CanonicalEducation",
    "CanonicalProfile",
    "CanonicalCandidate",
]
