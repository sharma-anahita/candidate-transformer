"""
profile.py — Social and professional profile link model.

Design rationale:
  Profile links serve two purposes in the pipeline:
    1. Display: show the candidate's online presence to recruiters
    2. Enrichment trigger: a GitHub URL found in a resume triggers the GitHub
       adapter, which then extracts skill signals from repos. This cross-adapter
       chaining is a key pipeline feature.

  ``Platform`` is an enum so the pipeline can dispatch to the right enrichment
  adapter without string matching. ``Platform.GITHUB`` → run GitHubAdapter.

  ``username`` is extracted and stored separately from ``url`` for deduplication:
  two different URL formats for the same username (http vs https, with/without
  trailing slash, /in/ vs /pub/profile/) must be collapsed to one profile.

  ``follower_count`` and ``public_repos`` are GitHub-specific but defined at
  this level (rather than in a GitHub-only submodel) to avoid over-engineering
  the model hierarchy before it is needed. When a LinkedIn adapter is built,
  it can use ``follower_count`` for connections. Other platforms leave these None.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Platform(str, Enum):
    """
    Supported professional and social platforms.

    Each value corresponds to a potential adapter that can enrich data
    from that platform. The pipeline dispatcher uses this enum to route
    profile URLs to the correct adapter.
    """

    GITHUB = "github"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    STACKOVERFLOW = "stackoverflow"
    PORTFOLIO = "portfolio"    # Personal websites, Netlify, Vercel, etc.
    BLOG = "blog"              # Medium, Substack, dev.to, Hashnode
    DRIBBBLE = "dribbble"
    BEHANCE = "behance"
    KAGGLE = "kaggle"
    LEETCODE = "leetcode"
    OTHER = "other"


# Domain → Platform mapping for URL-based platform detection.
# Used by adapters that receive a generic URL column and need to classify it.
DOMAIN_TO_PLATFORM: dict[str, Platform] = {
    "github.com": Platform.GITHUB,
    "linkedin.com": Platform.LINKEDIN,
    "twitter.com": Platform.TWITTER,
    "x.com": Platform.TWITTER,
    "stackoverflow.com": Platform.STACKOVERFLOW,
    "kaggle.com": Platform.KAGGLE,
    "leetcode.com": Platform.LEETCODE,
    "dribbble.com": Platform.DRIBBBLE,
    "behance.net": Platform.BEHANCE,
}


def detect_platform(url: str) -> Platform:
    """
    Infer the Platform from a URL string.

    Checks the URL against known domain patterns. Returns Platform.OTHER
    when no match is found — preserves the URL without losing it.

    Args:
        url: The raw URL string to classify.

    Returns:
        The matching Platform enum value, or Platform.OTHER.
    """
    url_lower = url.lower()
    for domain, platform in DOMAIN_TO_PLATFORM.items():
        if domain in url_lower:
            return platform
    return Platform.OTHER


# ─────────────────────────────────────────────────────────────────────────────
# Profile model
# ─────────────────────────────────────────────────────────────────────────────


class Profile(BaseModel):
    """
    A single social or professional profile link with metadata.

    Fields
    ------
    platform:
        Which platform this profile belongs to. Used to:
          - Route to the correct enrichment adapter (GitHub, LinkedIn, etc.)
          - Deduplicate profiles (two GitHub URLs with same username = one profile)
          - Filter profiles for display (show only GitHub + LinkedIn in a compact view)

    url:
        Full URL, cleaned of trailing slashes for consistent comparison.
        Stored as str (not HttpUrl) because some profile URLs in resumes are
        malformed (missing https://, or using http) and must be preserved
        rather than rejected.

    username:
        Platform-specific handle extracted from the URL or explicitly provided.
        Used for deduplication: two different URL formats for the same username
        collapse to one profile record.
        Extraction from URL is handled by each adapter or the profile normaliser,
        not by this model.

    display_name:
        Name shown on the platform, which may differ from the candidate's
        legal name (e.g., a GitHub display name of "thelastjenkins" vs
        "Marcus Jenkins" on the resume).

    is_verified:
        Whether we have confirmed the profile URL is live and belongs to this
        candidate (e.g., by checking the GitHub API with the username).
        Always False at extraction time.

    follower_count:
        Number of followers on the platform. Primarily used for GitHub and
        LinkedIn (connections). A proxy for professional influence, used in
        optional talent-scoring extensions.

    public_repos:
        Number of public repositories on GitHub. A proxy for open-source
        activity level. Only meaningful for Platform.GITHUB.
    """

    platform: Platform
    url: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    is_verified: bool = False
    follower_count: Optional[int] = Field(default=None, ge=0)
    public_repos: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(populate_by_name=True)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        """
        Strip leading/trailing whitespace and trailing slashes from URL.

        Consistent URL format is required for deduplication:
          "https://github.com/janedoe/" == "https://github.com/janedoe"
        """
        return v.strip().rstrip("/")
