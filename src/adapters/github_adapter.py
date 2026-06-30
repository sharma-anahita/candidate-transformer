from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from src.adapters.base import AdapterError, BaseAdapter
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.profile import Platform, Profile
from src.models.provenance import SourceType
from src.models.skill import Skill


TECH_FILES = {
    "package.json",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "pom.xml",
    "Cargo.toml",
    "go.mod",
    "tailwind.config.js",
    "vite.config.ts",
}


@dataclass
class GitHubRepo:
    full_name: str
    name: str
    html_url: str | None = None
    description: str | None = None
    default_branch: str = "main"
    languages: dict[str, int] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    readme: str | None = None
    workflows: list[str] = field(default_factory=list)
    tree_paths: list[str] = field(default_factory=list)
    tech_files: dict[str, str] = field(default_factory=dict)


class GitHubClient:
    def __init__(self, token: str | None = None, session=None, api_url: str = "https://api.github.com"):
        self.session = session or requests.Session()
        self.api_url = api_url.rstrip("/")
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def get(self, path: str, *, params: dict[str, Any] | None = None, allow_404: bool = False) -> Any:
        response = self.session.get(f"{self.api_url}{path}", headers=self.headers, params=params, timeout=10)
        if allow_404 and response.status_code == 404:
            return None
        if response.status_code in {401, 403}:
            raise AdapterError("GitHubAdapter", path, "GitHub API auth or rate-limit failure.")
        if response.status_code >= 400:
            raise AdapterError("GitHubAdapter", path, f"GitHub API returned HTTP {response.status_code}.")
        return response.json()


class ProfileExtractor:
    def extract(self, candidate: ExtractedCandidate, profile: dict[str, Any]) -> None:
        login = _clean(profile.get("login"))
        url = _clean(profile.get("html_url"))
        name = _clean(profile.get("name"))

        candidate.full_name = name or login
        candidate.summary = _clean(profile.get("bio"))

        if profile.get("location"):
            candidate.location = Location(raw=str(profile["location"]))

        if url:
            candidate.profiles.append(
                Profile(
                    platform=Platform.GITHUB,
                    url=url,
                    username=login,
                    display_name=name,
                    is_verified=True,
                    follower_count=profile.get("followers"),
                    public_repos=profile.get("public_repos"),
                )
            )

        candidate.metadata["github_profile"] = {
            "login": login,
            "company": _clean(profile.get("company")),
            "blog": _clean(profile.get("blog")),
            "followers": profile.get("followers"),
            "following": profile.get("following"),
            "public_repos": profile.get("public_repos"),
        }


class RepoExtractor:
    def __init__(self, client: GitHubClient, max_repositories: int):
        self.client = client
        self.max_repositories = max_repositories

    def extract(self, username: str) -> list[GitHubRepo]:
        try:
            # Fetch a larger batch of repos (up to 100) to ensure we find highly starred ones
            data = self.client.get(
                f"/users/{username}/repos",
                params={"sort": "updated", "per_page": max(100, self.max_repositories)},
            )
            print("Repository list retrieved successfully.")
        except Exception as exc:
            print(f"Repository list retrieval failed: {exc}")
            raise exc

        # Sort by stargazers_count descending, then by updated_at descending
        sorted_data = sorted(
            data or [],
            key=lambda x: (x.get("stargazers_count", 0), x.get("updated_at", "")),
            reverse=True
        )
        
        # Take only the top max_repositories
        top_data = sorted_data[:self.max_repositories]
        
        repos = []
        for item in top_data:
            repos.append(
                GitHubRepo(
                    full_name=item["full_name"],
                    name=item["name"],
                    html_url=item.get("html_url"),
                    description=item.get("description"),
                    default_branch=item.get("default_branch") or "main",
                )
            )
        return repos


class ReadmeExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def extract(self, repo: GitHubRepo) -> None:
        try:
            data = self.client.get(f"/repos/{repo.full_name}/readme", allow_404=True)
            if data and data.get("encoding") == "base64":
                repo.readme = _b64(data.get("content", ""))
        except Exception as exc:
            print(f"README unavailable (rate limited) for {repo.full_name}: {exc}")


class LanguageExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def extract(self, repo: GitHubRepo) -> None:
        try:
            repo.languages = self.client.get(f"/repos/{repo.full_name}/languages", allow_404=True) or {}
        except Exception as exc:
            print(f"Languages unavailable for {repo.full_name}: {exc}")
            repo.languages = {}


class TopicExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def extract(self, repo: GitHubRepo) -> None:
        try:
            data = self.client.get(f"/repos/{repo.full_name}/topics", allow_404=True) or {}
            repo.topics = list(data.get("names") or [])
        except Exception as exc:
            print(f"Topics unavailable for {repo.full_name}: {exc}")
            repo.topics = []


class WorkflowExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def extract(self, repo: GitHubRepo) -> None:
        try:
            data = self.client.get(f"/repos/{repo.full_name}/actions/workflows", allow_404=True) or {}
            repo.workflows = [w["name"] for w in data.get("workflows", []) if w.get("name")]
        except Exception as exc:
            print(f"Workflows unavailable for {repo.full_name}: {exc}")
            repo.workflows = []


class TreeExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def extract(self, repo: GitHubRepo) -> None:
        try:
            data = self.client.get(
                f"/repos/{repo.full_name}/git/trees/{repo.default_branch}",
                params={"recursive": "1"},
                allow_404=True,
            ) or {}
            repo.tree_paths = [x["path"] for x in data.get("tree", []) if x.get("type") == "blob" and x.get("path")]

            for path in repo.tree_paths:
                filename = path.rsplit("/", 1)[-1]
                if filename in TECH_FILES:
                    try:
                        content = self.client.get(f"/repos/{repo.full_name}/contents/{path}", allow_404=True)
                        if content and content.get("encoding") == "base64":
                            repo.tech_files[filename] = _b64(content.get("content", ""))
                    except Exception as file_exc:
                        print(f"File {path} unavailable for {repo.full_name}: {file_exc}")
        except Exception as exc:
            print(f"File tree unavailable for {repo.full_name}: {exc}")


class TechnologyExtractor:
    def extract(self, repos: list[GitHubRepo]) -> list[Skill]:
        seen: dict[str, Skill] = {}
        for repo in repos:
            self._add(seen, repo.languages.keys(), "github languages")
            self._add(seen, repo.topics, f"{repo.full_name} topics")
            self._add(seen, repo.workflows, f"{repo.full_name} workflows")
            self._add(seen, self._from_readme(repo.readme), f"{repo.full_name} README")

            for filename, content in repo.tech_files.items():
                self._add(seen, self._from_file(filename, content), f"{repo.full_name}:{filename}")

        return list(seen.values())

    def _add(self, seen: dict[str, Skill], names, context: str) -> None:
        for name in names:
            clean = _skill_name(name)
            if clean:
                lower_name = clean.lower()
                if lower_name not in seen:
                    seen[lower_name] = Skill(name=clean, is_inferred=True, source_context=context, github_occurrence_count=1)
                else:
                    seen[lower_name].github_occurrence_count += 1

    def _from_readme(self, text: str | None) -> list[str]:
        if not text:
            return []
        known = ["React", "TypeScript", "JavaScript", "Python", "Django", "FastAPI", "Flask",
                 "Docker", "Kubernetes", "AWS", "PostgreSQL", "Redis", "Tailwind CSS", "Vite"]
        lower = text.lower()
        return [x for x in known if x.lower() in lower]

    def _from_file(self, filename: str, content: str) -> list[str]:
        if filename == "package.json":
            data = json.loads(content)
            deps = []
            for key in ("dependencies", "devDependencies"):
                deps.extend((data.get(key) or {}).keys())
            return ["Node.js", *deps]
        if filename == "requirements.txt":
            return [re.split(r"[<>=!~\[]", line.strip())[0] for line in content.splitlines() if line.strip()]
        if filename == "Dockerfile":
            return ["Docker"]
        if filename == "docker-compose.yml":
            return ["Docker Compose"]
        if filename == "pom.xml":
            return ["Java", "Maven", *re.findall(r"<artifactId>([^<]+)</artifactId>", content)]
        if filename == "Cargo.toml":
            return ["Rust"]
        if filename == "go.mod":
            return ["Go"]
        if filename == "tailwind.config.js":
            return ["Tailwind CSS"]
        if filename == "vite.config.ts":
            return ["Vite", "TypeScript"]
        return []


class GitHubAdapter(BaseAdapter[str]):
    def __init__(self, token: str | None = None, session=None, max_repositories: int = 10):
        self.client = GitHubClient(token=token, session=session)
        self.profile_extractor = ProfileExtractor()
        self.repo_extractor = RepoExtractor(self.client, max_repositories)
        self.readme_extractor = ReadmeExtractor(self.client)
        self.language_extractor = LanguageExtractor(self.client)
        self.topic_extractor = TopicExtractor(self.client)
        self.workflow_extractor = WorkflowExtractor(self.client)
        self.tree_extractor = TreeExtractor(self.client)
        self.technology_extractor = TechnologyExtractor()

    @property
    def source_type(self) -> SourceType:
        return SourceType.GITHUB

    def validate_source(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise AdapterError(self.adapter_name, "github", "Expected GitHub username or profile URL.")

    def _extract(self, source: str) -> ExtractedCandidate:
        username = _username(source)
        candidate = self._new_candidate(f"https://github.com/{username}")

        if self.client.headers.get("Authorization"):
            print("GitHub API token found; using authenticated mode.")
        else:
            print("Authentication unavailable; using public GitHub API.")

        profile = None
        try:
            profile = self.client.get(f"/users/{username}")
            print("GitHub profile retrieved successfully.")
        except Exception as exc:
            print(f"GitHub profile retrieval failed: {exc}")
            raise exc

        if profile:
            try:
                self.profile_extractor.extract(candidate, profile)
            except Exception as exc:
                print(f"GitHub profile parsing failed: {exc}")
                candidate.add_warning(field="github_api", message=f"Failed to parse profile details: {exc}")

        repos = []
        try:
            repos = self.repo_extractor.extract(username)
        except Exception as exc:
            print(f"Failed to extract repositories: {exc}")
            candidate.add_warning(field="github_api", message=f"Failed to retrieve repositories: {exc}")

        for repo in repos:
            try:
                self.language_extractor.extract(repo)
            except Exception as exc:
                candidate.add_warning(field="github_api", message=f"Failed to extract languages for {repo.name}: {exc}")

            try:
                self.topic_extractor.extract(repo)
            except Exception as exc:
                candidate.add_warning(field="github_api", message=f"Failed to extract topics for {repo.name}: {exc}")

            try:
                self.readme_extractor.extract(repo)
            except Exception as exc:
                candidate.add_warning(field="github_api", message=f"Failed to extract README for {repo.name}: {exc}")

            try:
                self.workflow_extractor.extract(repo)
            except Exception as exc:
                candidate.add_warning(field="github_api", message=f"Failed to extract workflows for {repo.name}: {exc}")

            try:
                self.tree_extractor.extract(repo)
            except Exception as exc:
                candidate.add_warning(field="github_api", message=f"Failed to extract tree files for {repo.name}: {exc}")

        try:
            candidate.skills = self.technology_extractor.extract(repos)
        except Exception as exc:
            print(f"Failed to extract skills: {exc}")
            candidate.add_warning(field="github_api", message=f"Failed to extract skills: {exc}")
            candidate.skills = []

        candidate.metadata["github_repositories"] = [
            {
                "name": r.name,
                "full_name": r.full_name,
                "url": r.html_url,
                "description": r.description,
                "languages": r.languages,
                "topics": r.topics,
                "workflows": r.workflows,
                "tech_files": sorted(r.tech_files) if r.tech_files else [],
            }
            for r in repos
        ]
        return candidate


def _username(source: str) -> str:
    raw = source.strip()
    if "github.com" not in raw.lower():
        return raw.strip("/")
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise AdapterError("GitHubAdapter", raw, "Missing GitHub username.")
    return parts[0]


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _b64(value: str) -> str:
    return base64.b64decode("".join(value.split())).decode("utf-8", errors="replace")


def _skill_name(value) -> Optional[str]:
    text = _clean(value)
    if not text:
        return None
    aliases = {"reactjs": "React", "nodejs": "Node.js", "node": "Node.js", "tailwindcss": "Tailwind CSS"}
    cleaned = text.strip("@").replace("_", " ").replace("-", " ")
    return aliases.get(cleaned.lower(), cleaned[:1].upper() + cleaned[1:])