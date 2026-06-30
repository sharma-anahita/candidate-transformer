"""
tests/adapters/test_section_detector.py — Unit tests for the SectionDetector.
"""

import pytest
from src.adapters.section_detector import SectionDetector


@pytest.fixture
def detector():
    return SectionDetector()


class TestSectionDetectorNormalization:
    def test_basic_normalization(self, detector):
        assert detector.normalize_line("  EXPERIENCE  ") == "experience"
        assert detector.normalize_line("Work Experience:") == "work experience"
        assert detector.normalize_line("Professional Experience -") == "professional experience"
        assert detector.normalize_line("technical skills |") == "technical skills"

    def test_collapse_multiple_spaces(self, detector):
        assert detector.normalize_line("academic   background") == "academic background"

    def test_strip_banner_decorations(self, detector):
        assert detector.normalize_line("--- EXPERIENCE ---") == "experience"
        assert detector.normalize_line("=== SKILLS ===") == "skills"
        assert detector.normalize_line("▬ PROJECTS ▬") == "projects"
        assert detector.normalize_line("*** EDUCATION ***") == "education"


class TestSectionDetectorHeadingDetection:
    def test_valid_headings(self, detector):
        assert detector._detect_heading("EXPERIENCE", "") == "experience"
        assert detector._detect_heading("Work Experience", "") == "experience"
        assert detector._detect_heading("academic background:", "") == "education"

    def test_heading_too_long(self, detector):
        # A heading must be <= 4 words
        assert detector._detect_heading("this is a very long heading text", "") is None

    def test_skills_subsection_exclusions(self, detector):
        # Outside skills, these could be detected (e.g. language)
        assert detector._detect_heading("languages", "") == "skills"
        # Inside skills, they should be ignored
        assert detector._detect_heading("languages", "skills") is None
        assert detector._detect_heading("frameworks", "skills") is None
        assert detector._detect_heading("tools", "skills") is None
        assert detector._detect_heading("databases", "skills") is None
        assert detector._detect_heading("operating systems", "skills") is None

    def test_projects_exclusions(self, detector):
        # Inside projects, matching 'project' or 'projects' again should be ignored
        assert detector._detect_heading("projects", "projects") is None
        assert detector._detect_heading("project", "projects") is None
        # Other headings (like Education) should still match
        assert detector._detect_heading("education", "projects") == "education"


class TestSectionDetectorSlicing:
    def test_full_resume_slicing(self, detector):
        resume_text = """John Doe
john.doe@gmail.com

SUMMARY
A passionate software engineer with 5 years experience.

EXPERIENCE
Software Engineer at Google
2020 - Present
- Built cool things

PROJECTS
MindTrack - AI Platform
- Architected React frontend

EDUCATION
MIT
B.S. Computer Science
2015 - 2019
"""
        sections = detector.detect(resume_text)

        assert "header" in sections
        assert "summary" in sections
        assert "experience" in sections
        assert "projects" in sections
        assert "education" in sections

        # Header section check (starts at line 0)
        assert sections["header"]["start_line"] == 0
        assert "John Doe" in sections["header"]["content"]
        assert "john.doe@gmail.com" in sections["header"]["content"]

        # Summary section check (excludes the heading line itself)
        assert "A passionate software engineer" in sections["summary"]["content"]
        assert "SUMMARY" not in sections["summary"]["content"]

        # Experience section check
        assert "Software Engineer at Google" in sections["experience"]["content"]
        assert "- Built cool things" in sections["experience"]["content"]

        # Projects section check
        assert "MindTrack - AI Platform" in sections["projects"]["content"]

        # Education section check
        assert "MIT" in sections["education"]["content"]
        assert "B.S. Computer Science" in sections["education"]["content"]

    def test_repeated_sections_merged(self, detector):
        resume_text = """John Doe
EXPERIENCE
Google Intern
PROJECTS
Project A
EXPERIENCE
Facebook Engineer
"""
        sections = detector.detect(resume_text)
        assert "experience" in sections
        assert "Google Intern" in sections["experience"]["content"]
        assert "Facebook Engineer" in sections["experience"]["content"]
