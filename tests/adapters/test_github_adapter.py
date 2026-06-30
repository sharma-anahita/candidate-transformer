import base64
import json

import pytest

from src.adapters.base import AdapterError
from src.adapters.github_adapter import GitHubAdapter
from src.models.provenance import SourceType


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
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


def b64(text):
    return base64.b64encode(text.encode()).decode()
def test_github_adapter_uses_mocked_rest_api(github_session):
    candidate = GitHubAdapter(session=github_session).extract("https://github.com/janedoe")

    assert candidate.source_type == SourceType.GITHUB
    assert candidate.full_name == "Jane Doe"
    assert candidate.profiles[0].username == "janedoe"
    assert "Python" in {skill.name for skill in candidate.skills}
    assert candidate.metadata["github_repositories"][0]["full_name"] == "janedoe/app"


def test_github_adapter_rejects_blank(github_session):
    with pytest.raises(AdapterError):
        GitHubAdapter(session=github_session).extract("")

def test_github_adapter_extracts_profile_repos_and_inferred_technologies():
    package_json = json.dumps({"dependencies": {"react": "^18.0.0"}, "devDependencies": {"vite": "^5.0.0"}})

    session = FakeSession(
        {
            "/users/janedoe": {
                "login": "janedoe",
                "name": "Jane Doe",
                "bio": "Builds developer tools",
                "location": "Austin, TX",
                "html_url": "https://github.com/janedoe",
                "followers": 42,
                "public_repos": 3,
            },
            "/users/janedoe/repos": [
                {
                    "name": "hr-app",
                    "full_name": "janedoe/hr-app",
                    "html_url": "https://github.com/janedoe/hr-app",
                    "description": "Hiring app",
                    "default_branch": "main",
                }
            ],
            "/repos/janedoe/hr-app/languages": {"Python": 1000, "TypeScript": 900},
            "/repos/janedoe/hr-app/topics": {"names": ["fastapi", "tailwindcss"]},
            "/repos/janedoe/hr-app/readme": {"encoding": "base64", "content": b64("Uses Docker and PostgreSQL")},
            "/repos/janedoe/hr-app/actions/workflows": {"workflows": [{"name": "CI"}]},
            "/repos/janedoe/hr-app/git/trees/main": {
                "tree": [
                    {"type": "blob", "path": "package.json"},
                    {"type": "blob", "path": "requirements.txt"},
                    {"type": "blob", "path": "Dockerfile"},
                ]
            },
            "/repos/janedoe/hr-app/contents/package.json": {"encoding": "base64", "content": b64(package_json)},
            "/repos/janedoe/hr-app/contents/requirements.txt": {"encoding": "base64", "content": b64("fastapi==0.111.0")},
            "/repos/janedoe/hr-app/contents/Dockerfile": {"encoding": "base64", "content": b64("FROM python:3.11")},
        }
    )

    candidate = GitHubAdapter(session=session).extract("https://github.com/janedoe")

    assert candidate.source_type == SourceType.GITHUB
    assert candidate.full_name == "Jane Doe"
    assert candidate.location.raw == "Austin, TX"
    assert candidate.profiles[0].username == "janedoe"

    skills = {s.name for s in candidate.skills}
    assert {"Python", "TypeScript", "Fastapi", "Tailwind CSS", "Docker", "PostgreSQL", "React", "Vite"} <= skills

    repo = candidate.metadata["github_repositories"][0]
    assert repo["full_name"] == "janedoe/hr-app"
    assert repo["tech_files"] == ["Dockerfile", "package.json", "requirements.txt"]


def test_github_adapter_rejects_empty_source():
    with pytest.raises(AdapterError):
        GitHubAdapter(session=FakeSession({})).extract("")


def test_github_adapter_best_effort_error_handling():
    session = FakeSession(
        {
            "/users/janedoe": {
                "login": "janedoe",
                "name": "Jane Doe",
                "bio": "Builds developer tools",
                "location": "Austin, TX",
                "html_url": "https://github.com/janedoe",
                "followers": 42,
                "public_repos": 3,
            },
        }
    )
    candidate = GitHubAdapter(session=session).extract("https://github.com/janedoe")
    assert candidate.full_name == "Jane Doe"
    assert any(w.field == "github_api" for w in candidate.warnings)


def test_github_adapter_repo_sorting():
    session = FakeSession(
        {
            "/users/janedoe": {
                "login": "janedoe",
                "name": "Jane Doe",
            },
            "/users/janedoe/repos": [
                {
                    "name": "recent-repo",
                    "full_name": "janedoe/recent-repo",
                    "stargazers_count": 0,
                    "updated_at": "2026-06-30T00:00:00Z",
                },
                {
                    "name": "highly-starred-repo",
                    "full_name": "janedoe/highly-starred-repo",
                    "stargazers_count": 100,
                    "updated_at": "2025-01-01T00:00:00Z",
                },
                {
                    "name": "old-repo",
                    "full_name": "janedoe/old-repo",
                    "stargazers_count": 10,
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ],
        }
    )
    adapter = GitHubAdapter(session=session, max_repositories=2)
    candidate = adapter.extract("https://github.com/janedoe")
    
    repos = candidate.metadata["github_repositories"]
    assert len(repos) == 2
    assert repos[0]["name"] == "highly-starred-repo"
    assert repos[1]["name"] == "old-repo"


def test_confidence_engine_github_name_and_matching_skills():
    from src.engines.confidence_engine import ConfidenceEngine
    from src.models.canonical_candidate import CanonicalCandidate, CanonicalSkill
    from src.models.provenance import Provenance, ExtractionMethod, ConfidenceField
    
    c = CanonicalCandidate(
        first_name=ConfidenceField(
            value="Jane",
            provenance=[
                Provenance(source_type=SourceType.GITHUB, source_id="gh1", method=ExtractionMethod.API_RESPONSE, adapter_name="GitHubAdapter")
            ],
            confidence=0.0
        ),
        last_name=ConfidenceField(
            value="Doe",
            provenance=[
                Provenance(source_type=SourceType.GITHUB, source_id="gh1", method=ExtractionMethod.API_RESPONSE, adapter_name="GitHubAdapter")
            ],
            confidence=0.0
        )
    )
    
    scored = ConfidenceEngine().score(c)
    assert scored.first_name.confidence == 0.0
    assert scored.last_name.confidence == 0.0
    
    c_skills = CanonicalCandidate(
        skills=[
            CanonicalSkill(
                name="Python",
                is_inferred=True,
                provenance=[
                    Provenance(source_type=SourceType.GITHUB, source_id="gh1", method=ExtractionMethod.REGEX, adapter_name="GitHubAdapter"),
                    Provenance(source_type=SourceType.RESUME, source_id="res1", method=ExtractionMethod.REGEX, adapter_name="ResumeAdapter")
                ]
            ),
            CanonicalSkill(
                name="Go",
                is_inferred=True,
                provenance=[
                    Provenance(source_type=SourceType.GITHUB, source_id="gh1", method=ExtractionMethod.REGEX, adapter_name="GitHubAdapter")
                ]
            )
        ]
    )
    
    scored_skills = ConfidenceEngine().score(c_skills)
    python_skill = [s for s in scored_skills.skills if s.name == "Python"][0]
    go_skill = [s for s in scored_skills.skills if s.name == "Go"][0]
    
    assert python_skill.confidence > go_skill.confidence


def test_github_adapter_profile_failure_raises_exception():
    session = FakeSession({})
    with pytest.raises(Exception):
        GitHubAdapter(session=session).extract("https://github.com/janedoe")