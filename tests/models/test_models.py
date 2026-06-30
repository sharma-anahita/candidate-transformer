"""
tests/models/test_models.py — Unit tests for all Pydantic models.

Coverage strategy:
  - Happy path: valid data produces expected model state.
  - Validation: invalid data raises the right Pydantic errors.
  - Computed fields: model_validators produce correct derived values.
  - Serialization: models round-trip through JSON correctly.
  - Properties: computed properties return expected values.
  - Generics: ConfidenceField[T] works for different T types.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.models.canonical_candidate import (
    CanonicalCandidate,
    CanonicalEmail,
    CanonicalPhone,
    CanonicalSkill,
)
from src.models.education import DegreeLevel, Education
from src.models.email import Email, EmailType
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate, ExtractionWarning
from src.models.location import Location
from src.models.normalized_candidate import NormalizationLog, NormalizedCandidate
from src.models.phone import Phone, PhoneType
from src.models.profile import Platform, Profile, detect_platform
from src.models.provenance import (
    ConfidenceField,
    ExtractionMethod,
    Provenance,
    SourceType,
)
from src.models.skill import ProficiencyLevel, Skill, SkillCategory


# ═════════════════════════════════════════════════════════════════════════════
# Provenance & ConfidenceField
# ═════════════════════════════════════════════════════════════════════════════


class TestProvenance:
    def test_happy_path(self):
        p = Provenance(
            source_type=SourceType.RESUME,
            adapter_name="ResumeAdapter",
            method=ExtractionMethod.REGEX,
            source_id="/path/to/resume.pdf",
            raw_value="jane.doe@gmail.com",
            confidence=0.9,
        )
        assert p.source_type == SourceType.RESUME
        assert p.confidence == 0.9
        assert p.raw_value == "jane.doe@gmail.com"

    def test_defaults(self):
        p = Provenance(
            source_type=SourceType.GITHUB,
            adapter_name="GitHubAdapter",
            method=ExtractionMethod.API_RESPONSE,
            source_id="https://github.com/janedoe",
        )
        assert p.confidence == 1.0
        assert p.extra == {}
        assert isinstance(p.extracted_at, datetime)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Provenance(
                source_type=SourceType.CSV,
                adapter_name="CSVAdapter",
                method=ExtractionMethod.STRUCTURED_FIELD,
                source_id="csv:row:1",
                confidence=1.5,  # > 1.0
            )

    def test_frozen(self):
        """Provenance is immutable after construction."""
        p = Provenance(
            source_type=SourceType.CSV,
            adapter_name="CSVAdapter",
            method=ExtractionMethod.STRUCTURED_FIELD,
            source_id="csv:row:1",
        )
        with pytest.raises(Exception):  # ValidationError or TypeError
            p.confidence = 0.5  # type: ignore

    def test_source_type_serialises_as_string(self):
        p = Provenance(
            source_type=SourceType.LINKEDIN,
            adapter_name="LinkedInAdapter",
            method=ExtractionMethod.API_RESPONSE,
            source_id="https://linkedin.com/in/janedoe",
        )
        data = json.loads(p.model_dump_json())
        assert data["source_type"] == "linkedin"

    def test_extra_metadata_preserved(self):
        p = Provenance(
            source_type=SourceType.GITHUB,
            adapter_name="GitHubAdapter",
            method=ExtractionMethod.API_RESPONSE,
            source_id="https://github.com/user",
            extra={"stars": 1200, "forks": 80},
        )
        assert p.extra["stars"] == 1200


class TestConfidenceField:
    def test_str_confidence_field(self):
        cf = ConfidenceField[str](value="Jane", confidence=0.95)
        assert cf.value == "Jane"
        assert cf.confidence == 0.95
        assert cf.is_inferred is False
        assert cf.conflicts == []

    def test_float_confidence_field(self):
        cf = ConfidenceField[float](value=3.8, confidence=0.80)
        assert cf.value == 3.8

    def test_with_provenance(self):
        prov = Provenance(
            source_type=SourceType.RESUME,
            adapter_name="ResumeAdapter",
            method=ExtractionMethod.REGEX,
            source_id="resume.pdf",
        )
        cf = ConfidenceField[str](value="Jane", confidence=0.9, provenance=[prov])
        assert len(cf.provenance) == 1
        assert cf.provenance[0].adapter_name == "ResumeAdapter"

    def test_inferred_field(self):
        cf = ConfidenceField[str](value="Python", confidence=0.65, is_inferred=True)
        assert cf.is_inferred is True

    def test_conflicts_stored(self):
        cf = ConfidenceField[str](
            value="San Francisco",
            confidence=0.8,
            conflicts=[{"source": "linkedin", "value": "Remote"}],
        )
        assert len(cf.conflicts) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Email
# ═════════════════════════════════════════════════════════════════════════════


class TestEmail:
    def test_normalises_to_lowercase(self):
        e = Email(address="JANE.DOE@GMAIL.COM")
        assert e.address == "jane.doe@gmail.com"

    def test_strips_whitespace(self):
        e = Email(address="  jane@example.com  ")
        assert e.address == "jane@example.com"

    def test_domain_derived_automatically(self):
        e = Email(address="jane@acmecorp.io")
        assert e.domain == "acmecorp.io"

    def test_explicit_domain_preserved(self):
        e = Email(address="jane@acmecorp.io", domain="custom.domain")
        assert e.domain == "custom.domain"

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            Email(address="not-an-email")

    def test_invalid_no_tld_raises(self):
        with pytest.raises(ValidationError):
            Email(address="jane@nodot")

    def test_email_type_default(self):
        e = Email(address="jane@example.com")
        assert e.type == EmailType.UNKNOWN

    def test_work_email_type(self):
        e = Email(address="jane@company.com", type=EmailType.WORK)
        assert e.type == EmailType.WORK

    def test_is_primary_default_false(self):
        e = Email(address="jane@example.com")
        assert e.is_primary is False

    def test_json_roundtrip(self):
        e = Email(address="jane@example.com", type=EmailType.PERSONAL)
        data = json.loads(e.model_dump_json())
        e2 = Email(**data)
        assert e2.address == e.address


# ═════════════════════════════════════════════════════════════════════════════
# Phone
# ═════════════════════════════════════════════════════════════════════════════


class TestPhone:
    def test_raw_stored_as_is(self):
        p = Phone(raw="+1 (415) 555-1234")
        assert p.raw == "+1 (415) 555-1234"

    def test_defaults(self):
        p = Phone(raw="555-1234")
        assert p.normalized is None
        assert p.country_code is None
        assert p.is_valid is False
        assert p.type == PhoneType.UNKNOWN

    def test_explicit_normalized(self):
        p = Phone(raw="+14155551234", normalized="+14155551234", is_valid=True)
        assert p.normalized == "+14155551234"
        assert p.is_valid is True

    def test_phone_type(self):
        p = Phone(raw="555-1234", type=PhoneType.MOBILE)
        assert p.type == PhoneType.MOBILE


# ═════════════════════════════════════════════════════════════════════════════
# Skill
# ═════════════════════════════════════════════════════════════════════════════


class TestSkill:
    def test_happy_path(self):
        s = Skill(name="Python")
        assert s.name == "Python"
        assert s.category == SkillCategory.UNKNOWN
        assert s.proficiency_level == ProficiencyLevel.UNKNOWN
        assert s.is_inferred is False

    def test_full_skill(self):
        s = Skill(
            name="FastAPI",
            aliases=["fast-api", "fastapi"],
            category=SkillCategory.FRAMEWORK,
            proficiency_level=ProficiencyLevel.ADVANCED,
            years_of_experience=3.5,
            last_used_year=2024,
            source_context="resume:skills_section",
        )
        assert s.aliases == ["fast-api", "fastapi"]
        assert s.years_of_experience == 3.5

    def test_proficiency_weight_expert(self):
        s = Skill(name="Python", proficiency_level=ProficiencyLevel.EXPERT)
        assert s.proficiency_weight() == 1.0

    def test_proficiency_weight_unknown(self):
        s = Skill(name="Python")
        assert s.proficiency_weight() == 0.0

    def test_inferred_skill(self):
        s = Skill(name="Django", is_inferred=True, source_context="requirements.txt")
        assert s.is_inferred is True

    def test_negative_years_invalid(self):
        with pytest.raises(ValidationError):
            Skill(name="Python", years_of_experience=-1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Experience
# ═════════════════════════════════════════════════════════════════════════════


class TestExperience:
    def test_happy_path(self):
        e = Experience(
            company="Acme Corp",
            title="Senior Engineer",
            start_date=date(2021, 1, 1),
            end_date=date(2024, 6, 1),
        )
        assert e.company == "Acme Corp"
        assert e.duration_months == 41

    def test_duration_computed(self):
        e = Experience(
            company="X",
            title="Engineer",
            start_date=date(2020, 3, 1),
            end_date=date(2022, 3, 1),
        )
        assert e.duration_months == 24

    def test_current_role_duration_uses_today(self):
        e = Experience(
            company="Current Co",
            title="Engineer",
            start_date=date(2023, 1, 1),
            is_current=True,
        )
        assert e.duration_months is not None
        assert e.duration_months > 0

    def test_is_current_inferred_from_raw_end_date(self):
        e = Experience(company="X", title="Eng", raw_end_date="Present")
        assert e.is_current is True

    def test_is_current_inferred_case_insensitive(self):
        for token in ["present", "PRESENT", "current", "now", "ongoing"]:
            e = Experience(company="X", title="Eng", raw_end_date=token)
            assert e.is_current is True, f"Failed for token: {token!r}"

    def test_no_dates_no_duration(self):
        e = Experience(company="X", title="Eng")
        assert e.duration_months is None

    def test_duration_floor_at_zero(self):
        """Duration never goes negative even with reversed dates."""
        e = Experience(
            company="X",
            title="Eng",
            start_date=date(2024, 6, 1),
            end_date=date(2024, 1, 1),  # end before start
        )
        assert e.duration_months == 0

    def test_technologies_stored(self):
        e = Experience(
            company="X",
            title="Eng",
            technologies=["Python", "FastAPI", "PostgreSQL"],
        )
        assert "FastAPI" in e.technologies


# ═════════════════════════════════════════════════════════════════════════════
# Education
# ═════════════════════════════════════════════════════════════════════════════


class TestEducation:
    def test_happy_path(self):
        ed = Education(
            institution="MIT",
            degree="B.S. Computer Science",
            degree_level=DegreeLevel.BACHELOR,
            field_of_study="Computer Science",
            gpa=3.9,
        )
        assert ed.institution == "MIT"
        assert ed.degree_level == DegreeLevel.BACHELOR
        assert ed.gpa == 3.9

    def test_gpa_exceeds_scale_raises(self):
        with pytest.raises(ValidationError):
            Education(institution="MIT", gpa=9.2, gpa_scale=4.0)

    def test_gpa_on_10_scale_valid(self):
        ed = Education(institution="IIT", gpa=9.2, gpa_scale=10.0)
        assert ed.gpa == 9.2

    def test_no_gpa_valid(self):
        ed = Education(institution="State U")
        assert ed.gpa is None

    def test_degree_level_defaults_unknown(self):
        ed = Education(institution="Unknown U")
        assert ed.degree_level == DegreeLevel.UNKNOWN

    def test_courses_stored(self):
        ed = Education(
            institution="UC Berkeley",
            courses=["Distributed Systems", "Machine Learning"],
        )
        assert "Machine Learning" in ed.courses


# ═════════════════════════════════════════════════════════════════════════════
# Location
# ═════════════════════════════════════════════════════════════════════════════


class TestLocation:
    def test_raw_only(self):
        loc = Location(raw="San Francisco, CA")
        assert loc.raw == "San Francisco, CA"
        assert loc.city is None

    def test_structured_location(self):
        loc = Location(city="Austin", state="Texas", country="United States", country_code="US")
        assert loc.display == "Austin, Texas, US"

    def test_display_falls_back_to_raw(self):
        loc = Location(raw="Greater Seattle Area")
        assert loc.display == "Greater Seattle Area"

    def test_display_prefers_state_code(self):
        loc = Location(city="Austin", state="Texas", state_code="TX", country_code="US")
        assert loc.display == "Austin, TX, US"

    def test_is_empty(self):
        loc = Location()
        assert loc.is_empty is True

    def test_not_empty_with_city(self):
        loc = Location(city="Paris")
        assert loc.is_empty is False

    def test_country_code_length_validation(self):
        with pytest.raises(ValidationError):
            Location(country_code="USA")  # must be 2 chars

    def test_latitude_bounds(self):
        with pytest.raises(ValidationError):
            Location(latitude=91.0)

    def test_longitude_bounds(self):
        with pytest.raises(ValidationError):
            Location(longitude=200.0)


# ═════════════════════════════════════════════════════════════════════════════
# Profile
# ═════════════════════════════════════════════════════════════════════════════


class TestProfile:
    def test_happy_path(self):
        p = Profile(platform=Platform.GITHUB, url="https://github.com/janedoe")
        assert p.platform == Platform.GITHUB
        assert p.url == "https://github.com/janedoe"

    def test_trailing_slash_stripped(self):
        p = Profile(platform=Platform.GITHUB, url="https://github.com/janedoe/")
        assert not p.url.endswith("/")

    def test_detect_platform_github(self):
        assert detect_platform("https://github.com/user") == Platform.GITHUB

    def test_detect_platform_linkedin(self):
        assert detect_platform("https://linkedin.com/in/user") == Platform.LINKEDIN

    def test_detect_platform_unknown(self):
        assert detect_platform("https://mypersonalsite.io") == Platform.OTHER

    def test_detect_platform_stackoverflow(self):
        assert detect_platform("https://stackoverflow.com/users/123") == Platform.STACKOVERFLOW


# ═════════════════════════════════════════════════════════════════════════════
# ExtractedCandidate
# ═════════════════════════════════════════════════════════════════════════════


class TestExtractedCandidate:
    def test_happy_path(self):
        c = ExtractedCandidate(
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
        )
        assert c.source_type == SourceType.CSV
        assert isinstance(c.extraction_id, UUID)
        assert c.first_name is None
        assert c.emails == []
        assert c.warnings == []

    def test_add_warning(self):
        c = ExtractedCandidate(
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
        )
        c.add_warning(field="phone", message="Cannot parse number", raw="555-CALL-NOW")
        assert c.has_warnings is True
        assert len(c.warnings) == 1
        assert c.warnings[0].field == "phone"
        assert c.warnings[0].raw == "555-CALL-NOW"

    def test_display_name_from_parts(self):
        c = ExtractedCandidate(
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
            first_name="Jane",
            middle_name="A.",
            last_name="Doe",
        )
        assert c.display_name == "Jane A. Doe"

    def test_display_name_from_full_name(self):
        c = ExtractedCandidate(
            source_type=SourceType.GITHUB,
            source_id="https://github.com/janedoe",
            adapter_name="GitHubAdapter",
            full_name="Jane Doe",
        )
        assert c.display_name == "Jane Doe"

    def test_display_name_parts_preferred_over_full_name(self):
        c = ExtractedCandidate(
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
            first_name="Jane",
            last_name="Doe",
            full_name="Jane Q. Doe",
        )
        assert c.display_name == "Jane Doe"

    def test_metadata_stored(self):
        c = ExtractedCandidate(
            source_type=SourceType.GITHUB,
            source_id="https://github.com/user",
            adapter_name="GitHubAdapter",
            metadata={"followers": 450, "public_repos": 32},
        )
        assert c.metadata["followers"] == 450

    def test_json_roundtrip(self):
        c = ExtractedCandidate(
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
            first_name="Jane",
            emails=[Email(address="jane@example.com")],
        )
        data = json.loads(c.model_dump_json())
        assert data["first_name"] == "Jane"
        assert data["emails"][0]["address"] == "jane@example.com"


# ═════════════════════════════════════════════════════════════════════════════
# NormalizedCandidate
# ═════════════════════════════════════════════════════════════════════════════


class TestNormalizedCandidate:
    def _make(self) -> NormalizedCandidate:
        from uuid import uuid4
        return NormalizedCandidate(
            extraction_id=uuid4(),
            source_type=SourceType.CSV,
            source_id="csv:row:1",
            adapter_name="CSVAdapter",
        )

    def test_happy_path(self):
        nc = self._make()
        assert nc.normalization_version == "1.0.0"
        assert nc.normalization_logs == []

    def test_log_appends(self):
        nc = self._make()
        nc.log(
            field="phones[0].normalized",
            normalizer="PhoneNormalizer",
            original="+1 (415) 555-1234",
            result="+14155551234",
        )
        assert len(nc.normalization_logs) == 1
        log = nc.normalization_logs[0]
        assert log.normalizer == "PhoneNormalizer"
        assert log.success is True

    def test_failed_normalizations(self):
        nc = self._make()
        nc.log("phone", "PhoneNormalizer", "bad-phone", None, success=False, message="Parse failed")
        nc.log("email", "EmailNormalizer", "jane@example.com", "jane@example.com", success=True)
        failed = nc.failed_normalizations
        assert len(failed) == 1
        assert failed[0].field == "phone"

    def test_log_frozen(self):
        nc = self._make()
        nc.log("phone", "PhoneNormalizer", "raw", "+1...", success=True)
        log = nc.normalization_logs[0]
        with pytest.raises(Exception):
            log.field = "mutated"  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# CanonicalCandidate
# ═════════════════════════════════════════════════════════════════════════════


class TestCanonicalCandidate:
    def _make(self, **kwargs) -> CanonicalCandidate:
        return CanonicalCandidate(**kwargs)

    def test_empty_candidate(self):
        c = self._make()
        assert isinstance(c.candidate_id, UUID)
        assert c.overall_confidence == 0.0
        assert c.display_name is None
        assert c.primary_email is None

    def test_display_name(self):
        c = self._make(
            first_name=ConfidenceField[str](value="Jane", confidence=0.9),
            last_name=ConfidenceField[str](value="Doe", confidence=0.9),
        )
        assert c.display_name == "Jane Doe"

    def test_primary_email_from_flag(self):
        c = self._make(
            emails=[
                CanonicalEmail(address="jane@work.com", is_primary=False, confidence=0.9),
                CanonicalEmail(address="jane@gmail.com", is_primary=True, confidence=0.95),
            ]
        )
        assert c.primary_email == "jane@gmail.com"

    def test_primary_email_fallback_to_first(self):
        c = self._make(
            emails=[
                CanonicalEmail(address="jane@example.com", is_primary=False, confidence=0.9),
            ]
        )
        assert c.primary_email == "jane@example.com"

    def test_primary_phone_e164(self):
        c = self._make(
            phones=[
                CanonicalPhone(
                    raw="+14155551234",
                    normalized="+14155551234",
                    is_primary=True,
                    confidence=0.95,
                )
            ]
        )
        assert c.primary_phone == "+14155551234"

    def test_needs_review_below_threshold(self):
        c = self._make(overall_confidence=0.5)
        assert c.needs_review(threshold=0.7) is True

    def test_needs_review_above_threshold(self):
        c = self._make(overall_confidence=0.85)
        assert c.needs_review(threshold=0.7) is False

    def test_get_skill_names_sorted_by_confidence(self):
        c = self._make(
            skills=[
                CanonicalSkill(name="Docker", confidence=0.6),
                CanonicalSkill(name="Python", confidence=0.95),
                CanonicalSkill(name="React", confidence=0.75),
            ]
        )
        names = c.get_skill_names()
        assert names == ["Python", "React", "Docker"]

    def test_json_roundtrip(self):
        c = self._make(
            first_name=ConfidenceField[str](value="Jane", confidence=0.9),
            overall_confidence=0.85,
        )
        data = json.loads(c.model_dump_json())
        assert data["overall_confidence"] == 0.85
        assert data["first_name"]["value"] == "Jane"
