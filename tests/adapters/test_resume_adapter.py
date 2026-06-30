"""
tests/adapters/test_resume_adapter.py — Unit tests for ResumeAdapter.

Testing strategy
────────────────
The parser methods (_extract_name, _parse_experience_block, etc.) are tested
directly with controlled text strings — no PDF files required. This gives us:
  - Fast tests (no I/O)
  - Precise coverage of parsing logic
  - Easy to add cases for newly encountered resume formats

Integration tests mock _pdf_to_text() to return the sample resume text from
samples/sample_resume.txt, keeping the full-pipeline test deterministic.

Coverage:
  Section detection — various header formats, all-caps, decorated, prefixed
  Section splitting — multi-section text, header content, edge cases
  Name extraction — happy path, all-caps, middle name, title/email skipped
  Email extraction — single, multiple, deduplication, invalid skipped
  Phone extraction — US formats, international, deduplication
  Links extraction — github, linkedin, portfolio, deduplication
  Summary extraction — from section, from header paragraph
  Skills extraction — comma, bullet, categorised, deduplication
  Experience blocks — standard multi-line, pipe format, is_current, date parsing
  Education blocks — degree level, GPA, GPA scale, coursework
  Projects extraction — title, tech stack, bullets stored as Experience
  Date range parsing — month+year, year-only, Present, invalid
  Degree level detection — PhD, Master, Bachelor, Associate, Unknown
  Bullet classification — achievements vs responsibilities
  Source ID — path-based, bytes-based
  validate_source — missing file, wrong extension
  Never-fail contract — bad data produces warnings, not crashes
  Integration — full pipeline with mocked pdfplumber, source_type/adapter_name
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.base import AdapterError
from src.adapters.resume_adapter import (
    ResumeAdapter,
    _DATE_RANGE_RE,
    _EMAIL_RE,
    _PHONE_RE,
    _URL_RE,
)
from src.models.education import DegreeLevel
from src.models.extracted_candidate import ExtractedCandidate
from src.models.profile import Platform
from src.models.provenance import SourceType


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """\
Jane A. Doe
jane.doe@gmail.com | +14155551234 | https://github.com/janedoe | https://linkedin.com/in/jane-doe
San Francisco, CA

SUMMARY
Senior backend engineer with 8 years of experience designing distributed systems.
Expert in Python, FastAPI, and cloud-native architectures on AWS and GCP.

SKILLS
Languages: Python, Go, SQL, Bash
Frameworks: FastAPI, Django, Flask
Databases: PostgreSQL, Redis

EXPERIENCE

Senior Software Engineer
Acme Corporation, San Francisco, CA
Jan 2022 – Present
• Led the development of a microservices platform serving 10M+ monthly active users
• Reduced API p99 latency by 40% through Redis caching
• Mentored 5 junior engineers

Software Engineer
Beta Inc, New York, NY
Mar 2019 – Dec 2021
• Built the core recommendation engine improving CTR by 22% across 3M users
• Migrated monolithic Rails application to microservices

EDUCATION

University of California, Berkeley
Bachelor of Science in Computer Science
2015 – 2019
GPA: 3.9/4.0
Relevant Coursework: Distributed Systems, Machine Learning

PROJECTS

Distributed Rate Limiter
An open-source token-bucket rate limiter implemented in Go with Redis backend.
Tech: Go, Redis, Docker
• Achieved 200K requests/sec throughput
• 1.2K GitHub stars
"""


@pytest.fixture
def adapter() -> ResumeAdapter:
    return ResumeAdapter()


@pytest.fixture
def sample_candidate(adapter: ResumeAdapter) -> ExtractedCandidate:
    """Full candidate extracted from SAMPLE_RESUME_TEXT via mocked pdfplumber."""
    with patch.object(adapter, "_pdf_to_text", return_value=SAMPLE_RESUME_TEXT):
        # Create a fake PDF path so validate_source can be bypassed
        with patch.object(adapter, "validate_source", return_value=None):
            return adapter.extract(Path("fake_resume.pdf"))


# ═════════════════════════════════════════════════════════════════════════════
# Module-level regex patterns (sanity checks)
# ═════════════════════════════════════════════════════════════════════════════


class TestRegexPatterns:
    def test_email_re_matches_standard(self):
        assert _EMAIL_RE.search("jane.doe@gmail.com")

    def test_email_re_matches_plus_addressing(self):
        assert _EMAIL_RE.search("jane+work@example.co.uk")

    def test_email_re_no_match_no_tld(self):
        assert not _EMAIL_RE.fullmatch("jane@nodot")

    def test_phone_re_us_dashes(self):
        assert _PHONE_RE.search("+1 415-555-1234")

    def test_phone_re_us_parentheses(self):
        assert _PHONE_RE.search("(415) 555-1234")

    def test_phone_re_international(self):
        assert _PHONE_RE.search("+44 7911 123456")

    def test_phone_re_no_match_year_range(self):
        # "2019-2022" should NOT match as a phone number
        match = _PHONE_RE.search("2019-2022")
        assert match is None

    def test_url_re_https(self):
        assert _URL_RE.search("https://github.com/janedoe")

    def test_url_re_bare_github(self):
        assert _URL_RE.search("github.com/janedoe")

    def test_date_range_re_month_year(self):
        assert _DATE_RANGE_RE.search("Jan 2022 – Present")

    def test_date_range_re_year_only(self):
        assert _DATE_RANGE_RE.search("2019 - 2022")

    def test_date_range_re_present_variants(self):
        for variant in ["Present", "Current", "Now", "Ongoing"]:
            assert _DATE_RANGE_RE.search(f"Jan 2020 – {variant}"), f"Failed for {variant!r}"

    def test_date_range_re_full_month_names(self):
        assert _DATE_RANGE_RE.search("March 2018 – December 2020")


# ═════════════════════════════════════════════════════════════════════════════
# Section header detection
# ═════════════════════════════════════════════════════════════════════════════


class TestSectionHeaderDetection:
    def test_experience_all_caps(self, adapter):
        assert adapter._detect_section_header("EXPERIENCE") == "experience"

    def test_experience_title_case(self, adapter):
        assert adapter._detect_section_header("Experience") == "experience"

    def test_work_history(self, adapter):
        assert adapter._detect_section_header("Work History") == "experience"

    def test_education(self, adapter):
        assert adapter._detect_section_header("EDUCATION") == "education"

    def test_skills(self, adapter):
        assert adapter._detect_section_header("Technical Skills") == "skills"

    def test_projects(self, adapter):
        assert adapter._detect_section_header("Personal Projects") == "projects"

    def test_summary(self, adapter):
        assert adapter._detect_section_header("Professional Summary") == "summary"

    def test_objective(self, adapter):
        assert adapter._detect_section_header("Objective") == "summary"

    def test_certifications(self, adapter):
        assert adapter._detect_section_header("CERTIFICATIONS") == "certifications"

    def test_decorated_header_dashes(self, adapter):
        assert adapter._detect_section_header("--- EXPERIENCE ---") == "experience"

    def test_decorated_header_equals(self, adapter):
        assert adapter._detect_section_header("=== SKILLS ===") == "skills"

    def test_regular_line_returns_none(self, adapter):
        assert adapter._detect_section_header("Led the development of microservices") is None

    def test_long_line_returns_none(self, adapter):
        assert adapter._detect_section_header("x" * 70) is None

    def test_empty_line_returns_none(self, adapter):
        assert adapter._detect_section_header("") is None

    def test_blank_line_returns_none(self, adapter):
        assert adapter._detect_section_header("   ") is None

    def test_all_decoration_returns_none(self, adapter):
        assert adapter._detect_section_header("---===---") is None

    def test_profile_header(self, adapter):
        assert adapter._detect_section_header("Profile") == "summary"


# ═════════════════════════════════════════════════════════════════════════════
# Section splitting
# ═════════════════════════════════════════════════════════════════════════════


class TestSectionSplitting:
    def test_header_section_captured(self, adapter):
        lines = ["Jane Doe", "jane@example.com", "", "EXPERIENCE", "Company A"]
        sections = adapter._split_into_sections(lines)
        assert "header" in sections
        assert any("Jane Doe" in ln for ln in sections["header"])

    def test_experience_section_captured(self, adapter):
        lines = ["Jane Doe", "", "EXPERIENCE", "Company A", "Jan 2020 – Present"]
        sections = adapter._split_into_sections(lines)
        assert "experience" in sections

    def test_multiple_sections(self, adapter):
        lines = [
            "Jane Doe", "",
            "EXPERIENCE", "Company A", "",
            "EDUCATION", "MIT", "",
            "SKILLS", "Python",
        ]
        sections = adapter._split_into_sections(lines)
        assert "experience" in sections
        assert "education" in sections
        assert "skills" in sections

    def test_section_content_not_mixed(self, adapter):
        lines = [
            "EXPERIENCE", "Company A",
            "",
            "EDUCATION", "MIT",
        ]
        sections = adapter._split_into_sections(lines)
        exp_text = "\n".join(sections.get("experience", []))
        assert "MIT" not in exp_text

    def test_empty_lines_in_input(self, adapter):
        lines = ["", "", "Jane Doe", "", ""]
        sections = adapter._split_into_sections(lines)
        assert "header" in sections


# ═════════════════════════════════════════════════════════════════════════════
# Name extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestNameExtraction:
    def test_simple_name(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["Jane Doe", "jane@example.com"])
        assert c.full_name == "Jane Doe"
        assert c.first_name == "Jane"
        assert c.last_name == "Doe"

    def test_name_with_middle_initial(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["Jane A. Doe", "jane@example.com"])
        assert c.first_name == "Jane"
        assert c.middle_name == "A."
        assert c.last_name == "Doe"

    def test_all_caps_name_normalised(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["JANE DOE", "jane@example.com"])
        assert c.full_name == "Jane Doe"

    def test_email_line_skipped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["jane@example.com", "Jane Doe"])
        assert c.full_name == "Jane Doe"

    def test_title_keyword_line_skipped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["Senior Software Engineer", "Jane Doe"])
        assert c.full_name == "Jane Doe"

    def test_section_header_skipped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["EXPERIENCE", "Jane Doe", "jane@x.com"])
        assert c.full_name == "Jane Doe"

    def test_url_line_skipped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["https://github.com/janedoe", "Jane Doe"])
        assert c.full_name == "Jane Doe"

    def test_year_containing_line_skipped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["Jan 2022 – Present", "Jane Doe"])
        assert c.full_name == "Jane Doe"

    def test_no_name_adds_warning(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["EXPERIENCE", "Software Engineer | Acme Corp | Jan 2020 – Present"])
        assert c.has_warnings
        assert any(w.field == "full_name" for w in c.warnings)

    def test_hyphenated_last_name(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_name(c, ["Mary-Jane Watson", "mary@example.com"])
        assert c.full_name is not None


# ═════════════════════════════════════════════════════════════════════════════
# Email extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestEmailExtraction:
    def test_single_email(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "Contact: jane@example.com")
        assert len(c.emails) == 1
        assert c.emails[0].address == "jane@example.com"

    def test_first_email_is_primary(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "jane@work.com | jane@gmail.com")
        assert c.emails[0].is_primary is True
        assert c.emails[1].is_primary is False

    def test_multiple_emails(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "jane@example.com, jane@work.com")
        assert len(c.emails) == 2

    def test_email_lowercased(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "JANE@EXAMPLE.COM")
        assert c.emails[0].address == "jane@example.com"

    def test_duplicate_email_deduplicated(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "jane@example.com jane@example.com")
        assert len(c.emails) == 1

    def test_no_emails_in_text(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_emails(c, "Jane Doe, Software Engineer, San Francisco")
        assert c.emails == []


# ═════════════════════════════════════════════════════════════════════════════
# Phone extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestPhoneExtraction:
    def test_us_with_country_code(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "+14155551234")
        assert len(c.phones) == 1
        assert c.phones[0].raw == "+14155551234"

    def test_us_parentheses_format(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "(415) 555-1234")
        assert len(c.phones) >= 1

    def test_us_dashes_format(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "415-555-1234")
        assert len(c.phones) >= 1

    def test_international_format(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "+44 7911 123456")
        assert len(c.phones) >= 1

    def test_first_phone_is_primary(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "+14155551234 | +14155559876")
        assert c.phones[0].is_primary is True

    def test_duplicate_phone_deduplicated(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "+14155551234 | +14155551234")
        assert len(c.phones) == 1

    def test_year_range_not_matched_as_phone(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_phones(c, "Experience: 2019 - 2022")
        assert len(c.phones) == 0


# ═════════════════════════════════════════════════════════════════════════════
# Links / profile extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestLinksExtraction:
    def test_github_classified(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "https://github.com/janedoe")
        assert any(p.platform == Platform.GITHUB for p in c.profiles)

    def test_linkedin_classified(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "https://linkedin.com/in/jane-doe")
        assert any(p.platform == Platform.LINKEDIN for p in c.profiles)

    def test_unknown_url_classified_as_other(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "https://myportfolio.io/work")
        assert any(p.platform == Platform.OTHER for p in c.profiles)

    def test_duplicate_urls_deduplicated(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "https://github.com/jane https://github.com/jane")
        assert len(c.profiles) == 1

    def test_trailing_punctuation_stripped(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "See https://github.com/jane.")
        assert c.profiles[0].url == "https://github.com/jane"

    def test_multiple_platforms_extracted(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_links(c, "https://github.com/jane | https://linkedin.com/in/jane")
        platforms = {p.platform for p in c.profiles}
        assert Platform.GITHUB in platforms
        assert Platform.LINKEDIN in platforms


# ═════════════════════════════════════════════════════════════════════════════
# Summary extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestSummaryExtraction:
    def test_from_summary_section(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "summary": [
                "Senior backend engineer with 8 years of experience.",
                "Expert in Python and distributed systems.",
            ]
        }
        adapter._extract_summary(c, sections, [])
        assert "8 years" in c.summary

    def test_empty_summary_section_uses_header(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "header": [
                "Jane Doe",
                "jane@example.com",
                "",
                "Passionate engineer specializing in scalable backend infrastructure and distributed systems.",
            ]
        }
        adapter._extract_summary(c, sections, [])
        assert c.summary is not None
        assert "Passionate" in c.summary

    def test_no_summary_leaves_none(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_summary(c, {"header": ["Jane Doe", "jane@example.com"]}, [])
        assert c.summary is None


# ═════════════════════════════════════════════════════════════════════════════
# Skills extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestSkillsExtraction:
    def test_comma_separated(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {"skills": ["Python, FastAPI, PostgreSQL"]})
        names = [s.name for s in c.skills]
        assert "Python" in names
        assert "FastAPI" in names

    def test_categorised_skills(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {"skills": [
            "Languages: Python, Go",
            "Databases: PostgreSQL, Redis",
        ]}
        adapter._extract_skills(c, sections)
        names = [s.name for s in c.skills]
        assert "Python" in names
        assert "PostgreSQL" in names

    def test_bullet_separated(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {"skills": ["Python • FastAPI • Docker"]})
        names = [s.name for s in c.skills]
        assert "Python" in names
        assert "FastAPI" in names

    def test_deduplication(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {"skills": ["Python, Python, python"]})
        python_count = sum(1 for s in c.skills if s.name.lower() == "python")
        assert python_count == 1

    def test_short_noise_filtered(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {"skills": ["a, Python, 123"]})
        names = [s.name for s in c.skills]
        assert "a" not in names
        assert "123" not in names
        assert "Python" in names

    def test_no_skills_section_leaves_empty(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {})
        assert c.skills == []

    def test_source_context_set(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        adapter._extract_skills(c, {"skills": ["Python"]})
        assert c.skills[0].source_context == "resume:skills_section"


# ═════════════════════════════════════════════════════════════════════════════
# Experience block parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestExperienceBlockParsing:
    def test_standard_multiline_block(self, adapter):
        block = [
            "Senior Software Engineer",
            "Acme Corporation, San Francisco, CA",
            "Jan 2022 – Present",
            "• Led the development of microservices platform serving 10M+ users",
            "• Reduced API latency by 40%",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry is not None
        assert entry.title == "Senior Software Engineer"
        assert entry.company == "Acme Corporation"
        assert entry.is_current is True

    def test_year_range_block(self, adapter):
        block = [
            "Software Engineer",
            "Beta Inc",
            "2019 - 2021",
            "• Built React frontend",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry is not None
        assert entry.start_date == date(2019, 1, 1)
        assert entry.end_date == date(2021, 1, 1)
        assert entry.is_current is False

    def test_is_current_present(self, adapter):
        block = [
            "Lead Engineer",
            "TechCorp",
            "Mar 2022 – Present",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry.is_current is True
        assert entry.end_date is None

    def test_pipe_separated_format(self, adapter):
        block = ["Senior Engineer | Acme Corp | Jan 2022 – Present"]
        entry = adapter._parse_experience_block(block)
        assert entry is not None
        assert entry.is_current is True

    def test_no_date_returns_none(self, adapter):
        block = [
            "Software Engineer",
            "Acme Corp",
            "Led the team on important projects",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry is None

    def test_empty_block_returns_none(self, adapter):
        entry = adapter._parse_experience_block([])
        assert entry is None

    def test_achievements_detected(self, adapter):
        block = [
            "Engineer",
            "Company",
            "2020 - 2022",
            "• Reduced latency by 40%",
            "• Wrote unit tests",
        ]
        entry = adapter._parse_experience_block(block)
        assert any("40%" in a for a in entry.achievements)
        assert any("unit tests" in r for r in entry.responsibilities)

    def test_month_year_dates_parsed(self, adapter):
        block = [
            "Engineer",
            "Company",
            "Mar 2019 – Dec 2021",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry.start_date == date(2019, 3, 1)
        assert entry.end_date == date(2021, 12, 1)

    def test_raw_dates_stored(self, adapter):
        block = [
            "Engineer",
            "Company",
            "Jan 2022 – Present",
        ]
        entry = adapter._parse_experience_block(block)
        assert entry.raw_start_date == "Jan 2022"
        assert entry.raw_end_date == "Present"


# ═════════════════════════════════════════════════════════════════════════════
# Education block parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestEducationBlockParsing:
    def test_standard_block(self, adapter):
        block = [
            "University of California, Berkeley",
            "Bachelor of Science in Computer Science",
            "2015 – 2019",
            "GPA: 3.9/4.0",
        ]
        entry = adapter._parse_education_block(block)
        assert entry is not None
        assert entry.institution == "University of California, Berkeley"
        assert entry.degree_level == DegreeLevel.BACHELOR

    def test_gpa_extracted(self, adapter):
        block = [
            "MIT",
            "Master of Science in Computer Science",
            "2019 – 2021",
            "GPA: 4.0/4.0",
        ]
        entry = adapter._parse_education_block(block)
        assert entry.gpa == 4.0

    def test_gpa_scale_extracted(self, adapter):
        block = [
            "IIT Delhi",
            "Bachelor of Technology",
            "2012 – 2016",
            "GPA: 8.5/10.0",
        ]
        entry = adapter._parse_education_block(block)
        assert entry.gpa == 8.5
        assert entry.gpa_scale == 10.0

    def test_field_of_study_extracted(self, adapter):
        block = [
            "Stanford University",
            "Bachelor of Science in Computer Science",
            "2014 – 2018",
        ]
        entry = adapter._parse_education_block(block)
        assert entry.field_of_study == "Computer Science"

    def test_coursework_extracted(self, adapter):
        block = [
            "UC Berkeley",
            "B.S. Computer Science",
            "2015 – 2019",
            "Relevant Coursework: Distributed Systems, Machine Learning, Algorithms",
        ]
        entry = adapter._parse_education_block(block)
        assert "Distributed Systems" in entry.courses
        assert "Machine Learning" in entry.courses

    def test_phd_level_detected(self, adapter):
        block = [
            "Carnegie Mellon University",
            "Ph.D. in Computer Science",
            "2018 – 2024",
        ]
        entry = adapter._parse_education_block(block)
        assert entry.degree_level == DegreeLevel.DOCTORATE

    def test_master_level_detected(self, adapter):
        block = [
            "Columbia University",
            "M.S. in Computer Science",
            "2019 – 2021",
        ]
        entry = adapter._parse_education_block(block)
        assert entry.degree_level == DegreeLevel.MASTER

    def test_empty_block_returns_none(self, adapter):
        assert adapter._parse_education_block([]) is None

    def test_date_line_as_institution_skipped(self, adapter):
        # A block whose first line contains a date range is NOT an education entry
        block = ["Jan 2022 – Present", "Engineer"]
        entry = adapter._parse_education_block(block)
        assert entry is None


# ═════════════════════════════════════════════════════════════════════════════
# Projects extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestProjectsExtraction:
    def test_project_stored_as_experience(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "projects": [
                "Distributed Rate Limiter",
                "An open-source rate limiter in Go.",
                "Tech: Go, Redis, Docker",
                "• 1.2K GitHub stars",
                "",
                "FastAPI Scaffold",
                "CLI tool for FastAPI.",
                "• 3K downloads/month",
            ]
        }
        adapter._extract_projects(c, sections)
        project_titles = [e.title for e in c.projects]
        assert "Distributed Rate Limiter" in project_titles

    def test_technologies_extracted_from_tech_line(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "projects": [
                "My Project",
                "Description here.",
                "Tech: Python, FastAPI, Docker",
            ]
        }
        adapter._extract_projects(c, sections)
        assert len(c.projects) == 1
        assert "Python" in c.projects[0].technologies

    def test_company_set_to_personal_project(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {"projects": ["My App", "A great app."]}
        adapter._extract_projects(c, sections)
        assert c.projects[0].company == "Personal Project"

    def test_employment_type_set_to_project(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {"projects": ["My App", "Description."]}
        adapter._extract_projects(c, sections)
        assert c.projects[0].employment_type == "project"

    def test_wrapped_bullet_points_and_page_breaks(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "projects": [
                "Scalable Media Streaming Backend — Node JS, Express.js",
                "• Optimized MongoDB aggregation pipelines for high-volume media retrieval and",
                "efficient metadata processing.",
                "• Designed and implemented 15+ REST API endpoints supporting user",
                "uploads, playlists and subscriptions."
            ]
        }
        adapter._extract_projects(c, sections)
        assert len(c.projects) == 1
        proj = c.projects[0]
        # The wrapped lines should be merged
        assert len(proj.responsibilities) == 2
        assert "retrieval and efficient metadata processing." in proj.responsibilities[0]
        assert "supporting user uploads, playlists and subscriptions." in proj.responsibilities[1]
        # Technologies should be extracted and normalized
        assert "Node.js" in proj.technologies
        assert "MongoDB" in proj.technologies
        assert "REST APIs" in proj.technologies

    def test_bullet_style_project_titles(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "projects": [
                "• MindTrack — React.js, Node.js",
                "• Built a full-stack wellness platform.",
                "• LogGPT | GitHub | Live",
                "• Designed a distributed logger."
            ]
        }
        adapter._extract_projects(c, sections)
        assert len(c.projects) == 2
        assert c.projects[0].title == "MindTrack — React.js, Node.js"
        assert c.projects[1].title == "LogGPT | GitHub | Live"

    def test_github_live_metadata(self, adapter):
        c = ExtractedCandidate(source_type=SourceType.RESUME, source_id="x", adapter_name="ResumeAdapter")
        sections = {
            "projects": [
                "Shelly — Cross-Platform Shell",
                "GitHub | Live",
                "Tech Stack: C Plus Plus, Win32APIs",
                "• Built shell supporting IPC and redirection."
            ]
        }
        adapter._extract_projects(c, sections)
        assert len(c.projects) == 1
        proj = c.projects[0]
        assert "C++" in proj.technologies
        assert "GitHub | Live" in proj.description
        assert "Tech Stack: C Plus Plus, Win32APIs" in proj.description


# ═════════════════════════════════════════════════════════════════════════════
# Date range parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestDateRangeParsing:
    def test_year_range(self, adapter):
        start, rs, end, re_, current = adapter._parse_date_range("2019 – 2022")
        assert start == date(2019, 1, 1)
        assert end == date(2022, 1, 1)
        assert current is False

    def test_month_year_range(self, adapter):
        start, rs, end, re_, current = adapter._parse_date_range("Mar 2019 – Dec 2021")
        assert start == date(2019, 3, 1)
        assert end == date(2021, 12, 1)

    def test_present_sets_current(self, adapter):
        start, rs, end, re_, current = adapter._parse_date_range("Jan 2022 – Present")
        assert current is True
        assert end is None

    def test_current_variant(self, adapter):
        _, _, _, _, current = adapter._parse_date_range("2020 – Current")
        assert current is True

    def test_now_variant(self, adapter):
        _, _, _, _, current = adapter._parse_date_range("2021 – Now")
        assert current is True

    def test_raw_strings_stored(self, adapter):
        _, raw_start, _, raw_end, _ = adapter._parse_date_range("Jan 2022 – Present")
        assert raw_start == "Jan 2022"
        assert raw_end == "Present"

    def test_invalid_date_returns_none_start(self, adapter):
        start, _, _, _, _ = adapter._parse_date_range("invalid – Present")
        assert start is None


# ═════════════════════════════════════════════════════════════════════════════
# Single date parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestSingleDateParsing:
    def test_year_only(self, adapter):
        assert adapter._parse_single_date("2019") == date(2019, 1, 1)

    def test_month_year(self, adapter):
        assert adapter._parse_single_date("Mar 2019") == date(2019, 3, 1)

    def test_full_month_name(self, adapter):
        assert adapter._parse_single_date("January 2020") == date(2020, 1, 1)

    def test_present_returns_none(self, adapter):
        assert adapter._parse_single_date("Present") is None

    def test_empty_string_returns_none(self, adapter):
        assert adapter._parse_single_date("") is None

    def test_none_returns_none(self, adapter):
        assert adapter._parse_single_date(None) is None

    def test_garbage_returns_none(self, adapter):
        assert adapter._parse_single_date("not-a-date") is None


# ═════════════════════════════════════════════════════════════════════════════
# Degree level detection
# ═════════════════════════════════════════════════════════════════════════════


class TestDegreeLevelDetection:
    def test_phd(self, adapter):
        assert adapter._detect_degree_level("Ph.D. in Computer Science") == DegreeLevel.DOCTORATE

    def test_phd_no_dots(self, adapter):
        assert adapter._detect_degree_level("PhD Computer Science") == DegreeLevel.DOCTORATE

    def test_master_ms(self, adapter):
        assert adapter._detect_degree_level("M.S. in Machine Learning") == DegreeLevel.MASTER

    def test_master_mba(self, adapter):
        assert adapter._detect_degree_level("MBA, Finance") == DegreeLevel.MASTER

    def test_master_full(self, adapter):
        assert adapter._detect_degree_level("Master of Science in Data Science") == DegreeLevel.MASTER

    def test_bachelor_bs(self, adapter):
        assert adapter._detect_degree_level("B.S. Computer Science") == DegreeLevel.BACHELOR

    def test_bachelor_full(self, adapter):
        assert adapter._detect_degree_level("Bachelor of Science in Engineering") == DegreeLevel.BACHELOR

    def test_bachelor_btech(self, adapter):
        assert adapter._detect_degree_level("B.Tech in Computer Science") == DegreeLevel.BACHELOR

    def test_associate(self, adapter):
        assert adapter._detect_degree_level("Associate of Science") == DegreeLevel.ASSOCIATE

    def test_certificate(self, adapter):
        assert adapter._detect_degree_level("Professional Certificate in Data Science") == DegreeLevel.CERTIFICATE

    def test_unknown(self, adapter):
        assert adapter._detect_degree_level("Relevant Coursework: Algorithms") == DegreeLevel.UNKNOWN


# ═════════════════════════════════════════════════════════════════════════════
# Bullet classification
# ═════════════════════════════════════════════════════════════════════════════


class TestBulletClassification:
    def test_percentage_is_achievement(self, adapter):
        resps, achvs = adapter._classify_bullets(["Reduced latency by 40%"])
        assert len(achvs) == 1
        assert len(resps) == 0

    def test_dollar_amount_is_achievement(self, adapter):
        resps, achvs = adapter._classify_bullets(["Saved $2M in infrastructure costs"])
        assert len(achvs) == 1

    def test_headcount_is_achievement(self, adapter):
        resps, achvs = adapter._classify_bullets(["Mentored 5 engineers"])
        assert len(achvs) == 1

    def test_plain_responsibility(self, adapter):
        resps, achvs = adapter._classify_bullets(["Collaborated with cross-functional teams"])
        assert len(resps) == 1
        assert len(achvs) == 0

    def test_bullet_markers_stripped(self, adapter):
        resps, _ = adapter._classify_bullets(["• Collaborated with teams"])
        assert resps[0] == "Collaborated with teams"

    def test_empty_lines_skipped(self, adapter):
        resps, achvs = adapter._classify_bullets(["", "  ", "• Did something"])
        assert len(resps) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Block splitting utility
# ═════════════════════════════════════════════════════════════════════════════


class TestSplitIntoBlocks:
    def test_two_blocks(self, adapter):
        lines = ["A", "B", "", "C", "D"]
        blocks = adapter._split_into_blocks(lines)
        assert len(blocks) == 2
        assert blocks[0] == ["A", "B"]
        assert blocks[1] == ["C", "D"]

    def test_single_block_no_blank(self, adapter):
        lines = ["A", "B", "C"]
        blocks = adapter._split_into_blocks(lines)
        assert len(blocks) == 1

    def test_trailing_blank_lines_ignored(self, adapter):
        lines = ["A", "B", "", "", ""]
        blocks = adapter._split_into_blocks(lines)
        assert len(blocks) == 1

    def test_empty_input(self, adapter):
        assert adapter._split_into_blocks([]) == []


# ═════════════════════════════════════════════════════════════════════════════
# Source ID
# ═════════════════════════════════════════════════════════════════════════════


class TestSourceId:
    def test_path_source_id_uses_filename(self, adapter):
        sid = adapter._build_source_id(Path("/resumes/jane_doe_resume.pdf"))
        assert "jane_doe_resume.pdf" in sid

    def test_str_source_id_uses_filename(self, adapter):
        sid = adapter._build_source_id("/resumes/jane.pdf")
        assert "jane.pdf" in sid

    def test_bytes_source_id_uses_hash(self, adapter):
        content = b"fake pdf content"
        sid = adapter._build_source_id(content)
        assert sid.startswith("resume:bytes:")

    def test_bytes_source_id_deterministic(self, adapter):
        content = b"same content"
        assert adapter._build_source_id(content) == adapter._build_source_id(content)


# ═════════════════════════════════════════════════════════════════════════════
# validate_source
# ═════════════════════════════════════════════════════════════════════════════


class TestValidateSource:
    def test_nonexistent_file_raises(self, adapter):
        with pytest.raises(AdapterError):
            adapter.validate_source(Path("/no/such/file.pdf"))

    def test_wrong_extension_raises(self, adapter, tmp_path):
        doc = tmp_path / "resume.docx"
        doc.write_text("dummy")
        with pytest.raises(AdapterError):
            adapter.validate_source(doc)

    def test_valid_pdf_path_does_not_raise(self, adapter, tmp_path):
        pdf = tmp_path / "resume.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        adapter.validate_source(pdf)  # should not raise

    def test_bytes_source_does_not_raise(self, adapter):
        adapter.validate_source(b"%PDF-1.4 dummy")  # no validation for bytes


# ═════════════════════════════════════════════════════════════════════════════
# Integration — full pipeline with mocked pdfplumber
# ═════════════════════════════════════════════════════════════════════════════


class TestResumeAdapterIntegration:
    def test_source_type(self, sample_candidate):
        assert sample_candidate.source_type == SourceType.RESUME

    def test_adapter_name(self, sample_candidate):
        assert sample_candidate.adapter_name == "ResumeAdapter"

    def test_source_id_contains_filename(self, sample_candidate):
        assert "fake_resume.pdf" in sample_candidate.source_id

    def test_raw_text_stored(self, sample_candidate):
        assert sample_candidate.raw_text is not None
        assert "Jane" in sample_candidate.raw_text

    def test_name_extracted(self, sample_candidate):
        assert sample_candidate.full_name == "Jane A. Doe"
        assert sample_candidate.first_name == "Jane"
        assert sample_candidate.last_name == "Doe"

    def test_email_extracted(self, sample_candidate):
        assert any(e.address == "jane.doe@gmail.com" for e in sample_candidate.emails)

    def test_phone_extracted(self, sample_candidate):
        assert len(sample_candidate.phones) >= 1

    def test_github_profile_extracted(self, sample_candidate):
        assert any(p.platform == Platform.GITHUB for p in sample_candidate.profiles)

    def test_linkedin_profile_extracted(self, sample_candidate):
        assert any(p.platform == Platform.LINKEDIN for p in sample_candidate.profiles)

    def test_skills_extracted(self, sample_candidate):
        skill_names = [s.name for s in sample_candidate.skills]
        assert "Python" in skill_names
        assert "FastAPI" in skill_names

    def test_experience_extracted(self, sample_candidate):
        # Exclude project entries
        work_exp = [e for e in sample_candidate.experience if e.employment_type != "project"]
        assert len(work_exp) >= 2

    def test_current_role_detected(self, sample_candidate):
        current = [e for e in sample_candidate.experience if e.is_current and e.employment_type != "project"]
        assert len(current) >= 1
        assert current[0].company == "Acme Corporation"

    def test_education_extracted(self, sample_candidate):
        assert len(sample_candidate.education) >= 1
        edu = sample_candidate.education[0]
        assert "Berkeley" in edu.institution
        assert edu.degree_level == DegreeLevel.BACHELOR

    def test_gpa_extracted(self, sample_candidate):
        assert sample_candidate.education[0].gpa == 3.9

    def test_summary_extracted(self, sample_candidate):
        assert sample_candidate.summary is not None
        assert "engineer" in sample_candidate.summary.lower()

    def test_projects_stored_as_experience(self, sample_candidate):
        assert len(sample_candidate.projects) >= 1

    def test_empty_pdf_adds_warning(self, adapter, tmp_path):
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        with patch.object(adapter, "_pdf_to_text", return_value=""):
            with patch.object(adapter, "validate_source", return_value=None):
                candidate = adapter.extract(pdf)
        assert candidate.has_warnings
        assert any(w.field == "raw_text" for w in candidate.warnings)

    def test_pdfplumber_error_raises_adapter_error(self, adapter, tmp_path):
        pdf = tmp_path / "corrupt.pdf"
        pdf.write_bytes(b"not a pdf")
        with patch.object(
            adapter, "_pdf_to_text", side_effect=AdapterError("ResumeAdapter", "x", "parse failed")
        ):
            with patch.object(adapter, "validate_source", return_value=None):
                with pytest.raises(AdapterError):
                    adapter.extract(pdf)

    def test_no_warnings_for_clean_resume(self, sample_candidate):
        # A well-formed resume should not produce warnings
        assert not sample_candidate.has_warnings


def test_resume_adapter_groq_success(adapter):
    from unittest.mock import MagicMock, patch
    
    with patch.object(adapter, "_load_groq_key", return_value="fake_key"):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"full_name": "Jane Groq", "skills": [{"name": "Python"}]}'))
        ]
        
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            with patch.object(adapter, "_pdf_to_text", return_value="Jane Groq Python developer"):
                with patch.object(adapter, "validate_source", return_value=None):
                    candidate = adapter.extract("fake.pdf")
                    
                    assert candidate.full_name == "Jane Groq"
                    assert len(candidate.skills) == 1
                    assert candidate.skills[0].name == "Python"
                    assert not candidate.has_warnings


def test_resume_adapter_groq_retry_on_temporary_failure(adapter):
    from unittest.mock import MagicMock, patch
    from openai import RateLimitError
    
    with patch.object(adapter, "_load_groq_key", return_value="fake_key"):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='{"full_name": "Jane Retry", "skills": []}'))
            ]
            
            mock_client.chat.completions.create.side_effect = [
                RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body={}),
                mock_response
            ]
            mock_openai.return_value = mock_client
            
            with patch.object(adapter, "_pdf_to_text", return_value="Jane Retry developer"):
                with patch.object(adapter, "validate_source", return_value=None):
                    with patch("time.sleep", return_value=None):
                        candidate = adapter.extract("fake.pdf")
                        
                        assert candidate.full_name == "Jane Retry"
                        assert mock_client.chat.completions.create.call_count == 2
                        assert not candidate.has_warnings


def test_resume_adapter_groq_fallback_to_heuristic(adapter):
    from unittest.mock import MagicMock, patch
    
    with patch.object(adapter, "_load_groq_key", return_value="fake_key"):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Fatal connection failure")
            mock_openai.return_value = mock_client
            
            with patch.object(adapter, "_pdf_to_text", return_value="Jane Fallback\njane@example.com"):
                with patch.object(adapter, "validate_source", return_value=None):
                    candidate = adapter.extract("fake.pdf")
                    
                    assert candidate.full_name == "Jane Fallback"
                    assert candidate.has_warnings
                    assert any("Groq LLM extraction failed" in w.message for w in candidate.warnings)


def test_resume_adapter_groq_schema_injection(adapter):
    from unittest.mock import MagicMock, patch
    
    with patch.object(adapter, "_load_groq_key", return_value="fake_key"):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"full_name": "Jane Schema", "skills": []}'))
        ]
        
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            with patch.object(adapter, "_pdf_to_text", return_value="Jane Schema"):
                with patch.object(adapter, "validate_source", return_value=None):
                    candidate = adapter.extract("fake.pdf")
                    
                    assert candidate.full_name == "Jane Schema"
                    
                    args, kwargs = mock_client.chat.completions.create.call_args
                    system_msg = kwargs["messages"][0]["content"]
                    assert "properties" in system_msg
                    assert "first_name" in system_msg
                    assert "skills" in system_msg


def test_resume_adapter_groq_coercion(adapter):
    from unittest.mock import MagicMock, patch
    
    with patch.object(adapter, "_load_groq_key", return_value="fake_key"):
        mock_response = MagicMock()
        mock_content = {
            "full_name": "Jane Coerced",
            "experience": [
                {
                    "company": "DRDO",
                    "title": "Intern",
                    "location": {"raw": "Delhi, India"}
                }
            ],
            "projects": [
                {
                    "company": "Personal Project",
                    "title": "MindTrack",
                    "is_current": None
                }
            ],
            "profiles": [
                {"platform": "codolio", "url": "https://codolio.com/jane"},
                {"platform": "linkedin", "url": None},
                {"platform": "github", "url": "https://github.com/jane"}
            ]
        }
        import json
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(mock_content)))
        ]
        
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            with patch.object(adapter, "_pdf_to_text", return_value="Jane Coerced"):
                with patch.object(adapter, "validate_source", return_value=None):
                    candidate = adapter.extract("fake.pdf")
                    
                    assert candidate.full_name == "Jane Coerced"
                    assert candidate.experience[0].location == "Delhi, India"
                    assert candidate.projects[0].is_current is False
                    
                    assert len(candidate.profiles) == 1
                    assert candidate.profiles[0].platform == "github"
                    assert candidate.profiles[0].url == "https://github.com/jane"
                    assert not candidate.has_warnings
