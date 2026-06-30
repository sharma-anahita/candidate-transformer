"""
location.py — Geographic location model.

Design rationale:
  Location data enters the pipeline at wildly varying granularity:
    - GitHub: "San Francisco, CA"                (raw string, no structure)
    - Resume:  "123 Main St, Austin, TX 78701"   (full address)
    - LinkedIn: "Greater Seattle Area"            (region-level string)
    - ATS JSON: {"city": "New York", "country": "US"}  (structured)

  The Location model handles all of these. Fields are granular so downstream
  consumers (job-matching, tax jurisdiction, commute-distance calculators) can
  use exactly the level of detail they need.

  ``raw`` is always stored — the normaliser needs it for geocoding and re-parsing.
  ``country_code`` uses ISO 3166-1 alpha-2 (2-letter codes) because it is the
  most widely supported standard in HR systems and legal compliance layers.

  Lat/lon are optional enrichments from geocoding. They enable radius-based
  search queries in ATS systems ("show me candidates within 50 miles of HQ").
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Location(BaseModel):
    """
    A geographic location at any level of granularity.

    Fields
    ------
    raw:
        The original, unmodified location string from the source.
        Always stored so normalisers can re-process it without re-fetching.
        Acts as the fallback display value when structured fields are absent.

    street:
        Street address (number + street name). Rarely available in public
        profiles; more common in form submissions.

    city:
        City or town name. The most common level of granularity in resumes
        and professional profiles.

    state:
        Full state / province name (e.g., "California", "Ontario").
        The state_code field carries the abbreviated form.

    state_code:
        ISO 3166-2 subdivision code, e.g., ``"CA"``, ``"NY"``, ``"BC"``.
        Preferred over ``state`` for machine comparisons because free-text
        state names vary ("Calif.", "California", "CA").

    country:
        Full country name (e.g., "United States of America").
        Human-readable fallback when country_code is absent.

    country_code:
        ISO 3166-1 alpha-2 code, e.g., ``"US"``, ``"IN"``, ``"GB"``.
        Constrained to exactly 2 characters.
        Critical for:
          - Legal compliance (data residency, jurisdiction-aware processing)
          - Salary benchmarking (different markets, different currencies)
          - Interview scheduling (cross-border timezone resolution)

    postal_code:
        ZIP / postal code. Stored as a string (not int) because postal codes
        in many countries contain letters or leading zeros that int would destroy.

    latitude / longitude:
        WGS84 coordinates. Populated optionally by a geocoding step.
        Bounded at construction time to valid geographic ranges.
        Enable geospatial queries in ATS dashboards (radius search, commute time).

    is_remote:
        Candidate's preference for remote work. Not a geographic location per se,
        but always mentioned alongside location ("Remote OK", "Open to relocation").
        Stored here to keep all location-related preferences together.

    timezone:
        IANA timezone string, e.g., ``"America/Los_Angeles"``, ``"Asia/Kolkata"``.
        Derived from lat/lon or city name by the normaliser.
        Enables automated interview scheduling without timezone conversion bugs.

    Notes
    -----
    All fields are Optional because partial location data is extremely common.
    A Location with only ``raw="San Francisco"`` is still valid and useful.
    """

    raw: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    state_code: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    postal_code: Optional[str] = None
    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude",
    )
    is_remote: Optional[bool] = None
    timezone: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def display(self) -> str:
        """
        Best-effort human-readable location string for UI display.

        Builds a comma-separated string from the most specific fields available,
        falling back to ``raw`` if no structured fields are present.

        Examples:
            city=Austin, state_code=TX, country_code=US → "Austin, TX, US"
            raw="Greater Seattle Area"                   → "Greater Seattle Area"
        """
        parts = [
            p
            for p in [
                self.city,
                self.state_code or self.state,
                self.country_code or self.country,
            ]
            if p
        ]
        return ", ".join(parts) if parts else (self.raw or "")

    @property
    def is_empty(self) -> bool:
        """Returns True if no location information is available at all."""
        return all(
            v is None
            for v in [
                self.raw,
                self.street,
                self.city,
                self.state,
                self.state_code,
                self.country,
                self.country_code,
                self.postal_code,
                self.latitude,
                self.longitude,
            ]
        )
