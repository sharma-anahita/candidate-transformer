"""
src/adapters/section_detector.py — Deterministic Section Detector for Resumes.
"""

from __future__ import annotations

import re
from typing import Any, Optional

SECTION_HEADERS: dict[str, list[str]] = {
    "header": [
        "contact",
        "contact information",
        "personal information",
        "profile information",
        "personal details"
    ],
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "executive summary",
        "profile",
        "professional profile",
        "career profile",
        "objective",
        "career objective",
        "professional objective",
        "about",
        "about me",
        "overview",
        "introduction",
        "bio",
        "career overview"
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "career history",
        "work history",
        "industry experience",
        "internships",
        "internship",
        "internship experience",
        "professional background",
        "relevant experience"
    ],
    "projects": [
        "projects",
        "project",
        "personal projects",
        "academic projects",
        "professional projects",
        "selected projects",
        "featured projects",
        "research projects",
        "software projects",
        "major projects",
        "key projects",
        "capstone project",
        "capstone projects"
    ],
    "education": [
        "education",
        "academic background",
        "educational background",
        "education and training",
        "qualifications",
        "academic qualifications",
        "education qualification",
        "degrees",
        "degree",
        "schooling",
        "academics",
        "academic details"
    ],
    "skills": [
        "skills",
        "skill",
        "technical skills",
        "technical skill",
        "technical expertise",
        "technical competencies",
        "technologies",
        "technology stack",
        "tech stack",
        "core competencies",
        "competencies",
        "areas of expertise",
        "expertise",
        "professional skills",
        "programming languages",
        "languages",
        "frameworks",
        "libraries",
        "tools",
        "developer tools",
        "software",
        "software skills",
        "database",
        "databases",
        "platforms",
        "cloud",
        "cloud technologies",
        "operating systems"
    ],
    "certifications": [
        "certifications",
        "certification",
        "licenses",
        "license",
        "professional certifications",
        "courses",
        "coursework",
        "online courses",
        "certificates",
        "credentials"
    ],
    "achievements": [
        "achievements",
        "achievement",
        "awards",
        "award",
        "honors",
        "honours",
        "recognition",
        "accomplishments",
        "highlights",
        "milestones",
        "distinctions"
    ],
    "publications": [
        "publications",
        "publication",
        "research",
        "research publications",
        "papers",
        "journal papers",
        "conference papers",
        "articles",
        "patents",
        "books",
        "thesis",
        "dissertation"
    ],
    "leadership": [
        "leadership",
        "leadership experience",
        "positions of responsibility",
        "position of responsibility",
        "por",
        "leadership roles",
        "student leadership",
        "organizational experience"
    ],
    "volunteering": [
        "volunteer",
        "volunteering",
        "volunteer experience",
        "community service",
        "social work",
        "extracurricular activities",
        "activities",
        "clubs",
        "organizations"
    ],
    "languages_spoken": [
        "languages spoken",
        "spoken languages",
        "language proficiency",
        "language",
        "linguistic skills"
    ],
    "interests": [
        "interests",
        "hobbies",
        "personal interests",
        "activities and interests",
        "extracurricular interests"
    ],
    "references": [
        "references",
        "reference",
        "professional references",
        "referees"
    ]
}


class SectionDetector:
    """
    Redesigned resume section detector stage.
    """

    def __init__(self) -> None:
        self._headers = {k: set(v) for k, v in SECTION_HEADERS.items()}

    def normalize_line(self, line: str) -> str:
        """
        Normalize a heading line for comparison:
        - lowercase
        - strip whitespace and leading/trailing decorations (dashes, equals, etc.)
        - remove trailing punctuation like :, -, |
        - collapse multiple spaces
        """
        s = line.strip().lower()
        # Remove leading/trailing decorations (e.g. === SKILLS === or --- EXPERIENCE ---)
        s = re.sub(r"^[=\-_*•·|▬▪◦\s]+|[=\-_*•·|▬▪◦\s]+$", "", s)
        # Remove trailing punctuation
        s = re.sub(r"[:\-|\s]+$", "", s)
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _detect_heading(self, line: str, current_section: str) -> Optional[str]:
        """
        Detect if a line is a section heading.
        """
        norm = self.normalize_line(line)
        if not norm:
            return None

        # Short check: typical headings are <= 4 words
        if len(norm.split()) > 4:
            return None

        matched_section = None
        for canonical, variants in self._headers.items():
            if norm in variants:
                matched_section = canonical
                break

        if not matched_section:
            return None

        # Exclusion rules:
        # 1. Inside Skills, do not treat sub-heading categories as new section headings
        if current_section == "skills":
            sub_skills = {
                "languages", "frameworks", "tools", "databases", "cloud",
                "operating systems", "language", "libraries", "software",
                "platforms", "database"
            }
            if norm in sub_skills:
                return None

        # 2. Inside Projects, do not treat project headers as projects heading transition again
        if current_section == "projects":
            if matched_section == "projects":
                return None

        return matched_section

    def detect(self, raw_text: str) -> dict[str, dict[str, Any]]:
        """
        Slices the resume text into sections based on recognized headings.
        """
        lines = raw_text.splitlines()
        boundaries: list[tuple[str, int]] = [("header", 0)]
        current_section = "header"

        for i, line in enumerate(lines):
            # Only consider non-empty lines for heading transitions
            if not line.strip():
                continue
            detected = self._detect_heading(line, current_section)
            if detected and detected != current_section:
                current_section = detected
                boundaries.append((current_section, i))

        result: dict[str, dict[str, Any]] = {}
        for idx, (sec_name, start_idx) in enumerate(boundaries):
            end_idx = (
                boundaries[idx + 1][1] - 1
                if idx + 1 < len(boundaries)
                else len(lines) - 1
            )

            # Slicing: exclude the heading line itself from content
            content_start = start_idx
            if idx > 0 or self._detect_heading(lines[start_idx], ""):
                content_start = start_idx + 1

            sec_lines = lines[content_start : end_idx + 1]
            content = "\n".join(sec_lines)

            # If section repeated, merge them (applies if people have scattered sections)
            if sec_name in result:
                result[sec_name]["end_line"] = end_idx
                result[sec_name]["content"] += "\n" + content
            else:
                result[sec_name] = {
                    "start_line": start_idx,
                    "end_line": end_idx,
                    "content": content,
                }

        return result
