"""
tests/conftest.py — Shared pytest fixtures for all test modules.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timezone

import pytest

from src.models.education import Education
from src.models.email import Email
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.normalized_candidate import NormalizedCandidate
from src.models.phone import Phone
from src.models.provenance import SourceType
from src.models.skill import Skill


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit & App Scaffolding Upload Classes
# ─────────────────────────────────────────────────────────────────────────────

class Upload:
    def __init__(self, text: str):
        self._bytes = text.encode("utf-8")

    def getvalue(self):
        return self._bytes


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeGitHubSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        path = url.replace("https://api.github.com", "")
        self.calls.append((path, params))
        payload = self.routes.get(path)
        if payload is None:
            return FakeResponse(404, {})
        return FakeResponse(200, payload)


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Existing core fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def csv_upload():
    return Upload("first_name,last_name,email,skills\nJane,Doe,JANE@example.com,\"Python, JS\"\n")


@pytest.fixture
def ats_payload():
    return {
        "first_name": "Jane",
        "last_name": "Doe",
        "email": "jane@example.com",
        "skills": "Python, FastAPI",
    }


@pytest.fixture
def ats_upload(ats_payload):
    return Upload(json.dumps(ats_payload))


@pytest.fixture
def github_session():
    package_json = json.dumps({"dependencies": {"react": "^18.0.0"}})
    return FakeGitHubSession(
        {
            "/users/janedoe": {
                "login": "janedoe",
                "name": "Jane Doe",
                "bio": "Builds tools",
                "location": "Austin, TX",
                "html_url": "https://github.com/janedoe",
                "followers": 3,
                "public_repos": 1,
            },
            "/users/janedoe/repos": [
                {"name": "app", "full_name": "janedoe/app", "html_url": "https://github.com/janedoe/app"}
            ],
            "/repos/janedoe/app/languages": {"Python": 1},
            "/repos/janedoe/app/topics": {"names": ["fastapi"]},
            "/repos/janedoe/app/readme": {"encoding": "base64", "content": b64("React Docker")},
            "/repos/janedoe/app/actions/workflows": {"workflows": [{"name": "CI"}]},
            "/repos/janedoe/app/git/trees/main": {"tree": [{"type": "blob", "path": "package.json"}]},
            "/repos/janedoe/app/contents/package.json": {"encoding": "base64", "content": b64(package_json)},
        }
    )


@pytest.fixture
def extracted_candidate():
    return ExtractedCandidate(
        source_type=SourceType.MANUAL,
        source_id="manual:1",
        adapter_name="ManualAdapter",
        full_name="jane DOE",
        emails=[Email(address="JANE@example.com")],
        phones=[Phone(raw="(415) 555-2671")],
        location=Location(raw="San Francisco, CA, US"),
        skills=[Skill(name="js"), Skill(name="Python")],
        experience=[Experience(company="Acme Inc.", title="Engineer", raw_start_date="Jan 2020")],
        education=[Education(institution="State University", degree="Bachelor of Science")],
    )


@pytest.fixture
def normalized_candidate():
    return NormalizedCandidate(
        extraction_id=uuid4(),
        source_type=SourceType.ATS_JSON,
        source_id="ats:1",
        adapter_name="ATSJsonAdapter",
        first_name="Jane",
        last_name="Doe",
        emails=[Email(address="jane@example.com")],
        phones=[Phone(raw="4155552671", normalized="+14155552671", is_valid=True)],
        skills=[Skill(name="Python")],
        experience=[Experience(company="Acme", title="Engineer", raw_start_date="2020")],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV row fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def full_csv_row() -> dict[str, str]:
    """A complete CSV row with all common fields populated."""
    return {
        "first_name": "Jane",
        "last_name": "Doe",
        "email": "jane.doe@gmail.com",
        "phone": "+14155551234",
        "location": "San Francisco, CA",
        "skills": "Python, FastAPI, PostgreSQL",
        "summary": "Senior backend engineer with 7 years experience.",
        "github_url": "https://github.com/janedoe",
        "linkedin_url": "https://linkedin.com/in/jane-doe",
        "current_company": "Acme Corp",
        "current_title": "Senior Software Engineer",
    }


@pytest.fixture
def minimal_csv_row() -> dict[str, str]:
    """A CSV row with only the bare minimum fields (name + email)."""
    return {
        "first_name": "Alice",
        "last_name": "Zhang",
        "email": "alice.zhang@example.com",
    }


@pytest.fixture
def bad_contact_csv_row() -> dict[str, str]:
    """A CSV row with malformed email — should produce warnings, not crash."""
    return {
        "first_name": "Bob",
        "last_name": "Broken",
        "email": "not-a-valid-email",
        "phone": "+14155551234",
    }


@pytest.fixture
def structured_location_csv_row() -> dict[str, str]:
    """A CSV row with structured location columns (city, state, country)."""
    return {
        "first_name": "Carlos",
        "last_name": "Rivera",
        "email": "carlos@example.com",
        "city": "Austin",
        "state": "Texas",
        "country": "United States",
    }


@pytest.fixture
def multi_skill_csv_row() -> dict[str, str]:
    """Tests various multi-value delimiter styles for skills."""
    return {
        "first_name": "Dev",
        "last_name": "Tester",
        "email": "dev@example.com",
        "skills": "Python; Go; Rust",  # semicolon-delimited
    }


@pytest.fixture
def pipe_skill_csv_row() -> dict[str, str]:
    """Skills delimited by pipe character."""
    return {
        "first_name": "Dev",
        "last_name": "Tester",
        "email": "dev@example.com",
        "skills": "Python|React|AWS",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATS JSON fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def greenhouse_candidate() -> dict:
    """Realistic Greenhouse ATS candidate JSON."""
    return {
        "id": 98765,
        "first_name": "Jane",
        "last_name": "Doe",
        "email_addresses": [
            {"value": "jane.doe@gmail.com", "type": "personal"},
            {"value": "jane.doe@acmecorp.com", "type": "work"},
        ],
        "phone_numbers": [
            {"value": "+14155551234", "type": "mobile"},
        ],
        "addresses": [
            {"value": "San Francisco, CA, USA", "type": "home"},
        ],
        "website_addresses": [
            {"value": "https://github.com/janedoe", "type": "github"},
            {"value": "https://linkedin.com/in/jane-doe", "type": "linkedin"},
        ],
        "tags": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "application": {
            "current_employer": "Acme Corp",
            "current_title": "Senior Software Engineer",
            "resume_text": "Senior backend engineer with 7 years experience.",
        },
    }


@pytest.fixture
def lever_candidate() -> dict:
    """Realistic Lever ATS candidate JSON."""
    return {
        "id": "lever-abc123",
        "name": "John Smith",
        "headline": "Lead Frontend Engineer",
        "summary": "Full-stack engineer focused on frontend architecture.",
        "location": "New York, NY",
        "emails": ["john.smith@betainc.com"],
        "phones": [{"value": "+12125559876", "type": "mobile"}],
        "links": ["https://github.com/jsmith", "https://linkedin.com/in/john-smith"],
        "tags": ["JavaScript", "React", "TypeScript"],
    }


@pytest.fixture
def flat_ats_candidate() -> dict:
    """Simple flat ATS JSON — no nesting."""
    return {
        "id": "flat-001",
        "first_name": "Sara",
        "last_name": "Connor",
        "email": "sara.connor@example.com",
        "phone": "+15551234567",
        "location": "Los Angeles, CA",
        "summary": "Senior systems engineer.",
        "skills": "Python, Kubernetes, Terraform",
        "current_company": "Skynet Ltd",
        "current_title": "Principal Engineer",
        "linkedin_url": "https://linkedin.com/in/saraconnor",
        "github_url": "https://github.com/saraconnor",
    }