"""
adapters/resume_adapter.py — PDF resume parser.

Architecture
────────────
Parsing a resume is a multi-stage pipeline within the adapter:

  PDF bytes/path
      ↓
  pdfplumber  →  raw_text  (full document text, preserving line breaks)
      ↓
  _split_into_sections()  →  {section_name: section_text}
      ↓
  Field extractors (one per field group):
      _extract_name()        — heuristic: first title-case, non-contact line
      _extract_emails()      — regex
      _extract_phones()      — regex
      _extract_links()       — regex + platform detection
      _extract_summary()     — section text or header paragraph
      _extract_skills()      — section text, multi-delimiter split
      _extract_experience()  — block parser: date range as entry boundary
      _extract_education()   — block parser: degree/institution/GPA
      _extract_projects()    — block parser: similar to experience

Design decisions
────────────────
1. All parsing logic is in private instance methods, not a separate module.
   This keeps the adapter self-contained and independently testable by calling
   the private methods directly in tests (Python doesn't enforce privacy).

2. Section detection is purely text-based (header keywords + casing heuristics).
   We deliberately avoid font-size detection because it varies unpredictably
   across PDF rendering engines and pdfplumber's extraction accuracy.

3. Experience/education blocks are delimited by *blank lines*, not by date
   patterns. Blank-line-delimited chunks are more universally reliable across
   the enormous variation in resume formatting styles.

4. Date parsing uses python-dateutil for flexibility, with a custom default
   date (Jan 1 of 2000) to handle year-only strings like "2019".

5. The ``_parse_*_block()`` methods return None for blocks that don't match
   the expected pattern — they never raise. The caller filters out None values.

6. pdfplumber is imported lazily inside _pdf_to_text() so the module can be
   imported in environments where pdfplumber is not installed (e.g., unit tests
   that mock the method).
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import date
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from src.adapters.base import AdapterError, BaseAdapter
from src.models.education import DegreeLevel, Education
from src.models.email import Email
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.phone import Phone
from src.models.profile import Platform, Profile, detect_platform
from src.models.provenance import SourceType
from src.models.skill import Skill

# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Standard email address
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Phone: US 10-digit formats + E.164 international
# Lookarounds prevent matching date ranges like "2019-2022"
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    # US: optional +1, then (NXX) or NXX, then NXX-XXXX
    r"(?:\+?1[\s.\-]?)?"
    r"(?:\((?P<ac1>[2-9]\d{2})\)|(?P<ac2>[2-9]\d{2}))"
    r"[\s.\-]?"
    r"(?P<exch>[2-9]\d{2})"
    r"[\s.\-]?"
    r"(?P<num>\d{4})"
    r"|"
    # International E.164: +CC followed by 6-20 digits/spaces/dashes
    r"\+(?P<cc>\d{1,3})[\s.\-]?(?P<intl>[\d\s.\-]{6,20})"
    r")"
    r"(?!\d)",
)

# URLs — https://, github.com/user, linkedin.com/in/user, etc.
_URL_RE = re.compile(
    r"(?:https?://[^\s\n<>\"'()]+)"
    r"|(?:(?:github|linkedin|twitter|gitlab|kaggle|leetcode|behance|dribbble)"
    r"\.com/[^\s\n<>\"'()]+)",
    re.IGNORECASE,
)

# Date range — "Jan 2022 – Present", "2019 - 2022", "March 2018 to Present"
_DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+"
    r")?\d{4}"
    r"\s*(?:[^a-zA-Z0-9\s]+|to|\s+)\s*"
    r"(?:"
    r"(?:(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+"
    r")?\d{4})"
    r"|(?:Present|Current|Now|Ongoing)"
    r")",
    re.IGNORECASE,
)

# GPA: "GPA: 3.9 / 4.0", "GPA 3.8", "3.9/4.0 GPA"
_GPA_RE = re.compile(
    r"GPA[:\s]*(\d\.\d+)\s*(?:/\s*(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# Bullet point line prefix
_BULLET_PREFIX_RE = re.compile(r"^[•·▪◦○●\-\*\+]\s+")

# Job-title keywords — lines containing these are likely titles, not names
_TITLE_KEYWORDS: frozenset[str] = frozenset(
    {
        "engineer", "developer", "architect", "manager", "director", "analyst",
        "designer", "consultant", "specialist", "coordinator", "administrator",
        "scientist", "researcher", "officer", "lead", "head", "principal",
        "staff", "senior", "junior", "associate", "intern", "vp", "cto",
        "ceo", "cfo", "coo", "president", "founder", "co-founder",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Section header taxonomy
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, list[str]] = {
    "summary": [
        "summary", "professional summary", "profile", "about me", "about",
        "objective", "career objective", "professional objective",
        "overview", "executive summary", "professional profile",
        "career summary", "qualifications summary", "professional overview",
    ],
    "experience": [
        "experience", "work experience", "work history", "employment",
        "employment history", "professional experience", "career history",
        "positions held", "professional background", "relevant experience",
    ],
    "education": [
        "education", "academic background", "educational background",
        "academics", "academic history", "qualifications", "degrees",
        "academic qualifications", "educational qualifications",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "technologies", "tech stack", "tools & technologies", "tools",
        "technical expertise", "areas of expertise", "expertise",
        "programming languages", "languages & tools", "key skills",
        "hard skills", "technical proficiencies", "technical knowledge",
        "technology stack",
    ],
    "projects": [
        "projects", "personal projects", "side projects", "open source",
        "open-source", "notable projects", "key projects", "selected projects",
        "portfolio", "academic projects", "independent projects",
    ],
    "certifications": [
        "certifications", "certificates", "certification",
        "licenses & certifications", "awards & certifications",
        "professional certifications", "licenses",
    ],
    "volunteer": [
        "volunteer", "volunteering", "community involvement",
        "volunteer experience", "community service",
    ],
    "awards": [
        "awards", "achievements", "honors", "honours", "awards & recognition",
        "recognition", "accomplishments",
    ],
    "publications": [
        "publications", "papers", "research", "research & publications",
        "academic papers",
    ],
    "languages": [
        "languages spoken", "language proficiency", "spoken languages",
    ],
    "interests": [
        "interests", "hobbies", "activities", "personal interests",
    ],
}

# Degree level keyword mapping (checked in order — most specific first)
_DEGREE_MAP: list[tuple[DegreeLevel, list[str]]] = [
    (DegreeLevel.DOCTORATE, [
        "ph.d", "phd", "doctor of philosophy", "doctorate",
        "d.phil", "doctor of science", "sc.d", "d.sc",
    ]),
    (DegreeLevel.MASTER, [
        "m.s.", "m.s", " ms ", "msc", "m.sc", "master of science",
        "m.a.", "m.a", " ma ", "master of arts",
        "mba", "m.b.a", "master of business",
        "meng", "m.eng", "master of engineering",
        "master of", "master's", "masters",
    ]),
    (DegreeLevel.BACHELOR, [
        "b.s.", "b.s ", " bs ", "bsc", "b.sc", "bachelor of science",
        "b.a.", "b.a ", " ba ", "bachelor of arts",
        "b.e.", " be ", "b.tech", "btech", "bachelor of engineering",
        "bachelor of technology", "bachelor of",
        "bachelor's", "bachelors",
    ]),
    (DegreeLevel.ASSOCIATE, [
        "associate", "a.s.", "a.a.", "a.a.s.",
    ]),
    (DegreeLevel.BOOTCAMP, [
        "bootcamp", "boot camp", "coding bootcamp", "coding school",
    ]),
    (DegreeLevel.CERTIFICATE, [
        "certificate", "diploma", "nanodegree", "professional certificate",
    ]),
    (DegreeLevel.HIGH_SCHOOL, [
        "high school", "secondary school", "ged", "high school diploma",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# ResumeAdapter
# ─────────────────────────────────────────────────────────────────────────────


class ResumeAdapter(BaseAdapter[Union[Path, str, bytes]]):
    """
    Adapter for PDF resume files.

    Accepts: ``Path | str`` (file path) or ``bytes`` (raw PDF content,
    e.g., from a Streamlit file uploader or S3 download).

    Returns a single ExtractedCandidate with all available fields populated.
    Non-fatal parsing issues are recorded as ExtractionWarnings; the adapter
    never raises on bad data (only on unrecoverable file-level errors).

    Sections detected: summary, experience, education, skills, projects,
    certifications, volunteer, awards, publications, languages, interests.

    Limitations (by design for v1):
    - Scanned PDFs (image-only) will produce empty text + a warning.
    - Multi-column PDF layouts may confuse the line-based section splitter.
    - Non-English resumes are partially supported (section headers may not match).
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.RESUME

    # ── BaseAdapter interface ─────────────────────────────────────────────────

    def validate_source(self, source: Union[Path, str, bytes]) -> None:
        """Validate that the file exists and is a PDF (for path-based sources)."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise AdapterError(
                    self.adapter_name,
                    str(source),
                    f"File not found: {source}",
                )
            if path.suffix.lower() != ".pdf":
                raise AdapterError(
                    self.adapter_name,
                    str(source),
                    f"Expected a .pdf file, got: {path.suffix!r}",
                )

    def _coerce_json_by_schema(self, data: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> Any:
        def resolve_ref(sub_schema: dict[str, Any]) -> dict[str, Any]:
            if "$ref" in sub_schema:
                ref_path = sub_schema["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    return root_schema.get("$defs", {}).get(def_name, sub_schema)
                elif ref_path.startswith("#/definitions/"):
                    def_name = ref_path.split("/")[-1]
                    return root_schema.get("definitions", {}).get(def_name, sub_schema)
            return sub_schema

        resolved_schema = resolve_ref(schema)
        types = []
        raw_type = resolved_schema.get("type")
        is_nullable = False

        if isinstance(raw_type, list):
            types = raw_type
            if "null" in types:
                is_nullable = True
        elif isinstance(raw_type, str):
            types = [raw_type]

        if not types:
            for key in ("anyOf", "oneOf", "allOf"):
                if key in resolved_schema:
                    for sub in resolved_schema[key]:
                        sub_type = sub.get("type")
                        if isinstance(sub_type, str):
                            types.append(sub_type)
                            if sub_type == "null":
                                is_nullable = True
                        elif isinstance(sub_type, list):
                            types.extend(sub_type)
                            if "null" in sub_type:
                                is_nullable = True

        if data is None:
            if is_nullable:
                return None
            if "boolean" in types:
                return False
            if "integer" in types or "number" in types:
                return 0
            if "string" in types:
                return "N/A"
            if "array" in types:
                return []
            if "object" in types:
                return {}
            return None

        # 1. Coerce String type
        if "string" in types:
            if isinstance(data, list):
                data = "\n".join(str(x) for x in data)
            elif isinstance(data, dict):
                data = data.get("raw", str(data))
            elif data is None and not is_nullable:
                data = "N/A"
            elif data is not None:
                data = str(data)
            return data

        # 2. Coerce Boolean type
        if "boolean" in types:
            if data is None:
                if is_nullable:
                    return None
                return False
            if isinstance(data, str):
                return data.lower() in ("true", "1", "yes")
            return bool(data)

        # 3. Coerce Number/Integer type
        if "integer" in types or "number" in types:
            if data is None:
                if is_nullable:
                    return None
                return 0
            try:
                if "integer" in types:
                    return int(float(data))
                return float(data)
            except (ValueError, TypeError):
                return 0

        # 4. Coerce Array type
        if "array" in types:
            if data is None:
                if is_nullable:
                    return None
                return []
            if not isinstance(data, list):
                data = [data]

            if "items" in resolved_schema:
                item_schema = resolved_schema["items"]
                coerced_list = []
                for item in data:
                    coerced_list.append(self._coerce_json_by_schema(item, item_schema, root_schema))
                return coerced_list
            return data

        # 5. Coerce Object type
        if "object" in types:
            if data is None:
                if is_nullable:
                    return None
                data = {}
            if isinstance(data, str):
                data = {"raw": data}
            if not isinstance(data, dict):
                return data

            properties = resolved_schema.get("properties", {})
            required = resolved_schema.get("required", [])

            for req_prop in required:
                if req_prop not in data or data[req_prop] is None:
                    prop_schema = resolve_ref(properties.get(req_prop, {}))
                    prop_type = prop_schema.get("type", "string")
                    prop_nullable = False
                    if isinstance(prop_type, list) and "null" in prop_type:
                        prop_nullable = True
                    elif isinstance(prop_type, str) and prop_type == "null":
                        prop_nullable = True

                    if prop_nullable:
                        data[req_prop] = None
                    else:
                        if "array" in prop_type:
                            data[req_prop] = []
                        elif "boolean" in prop_type:
                            data[req_prop] = False
                        elif "integer" in prop_type or "number" in prop_type:
                            data[req_prop] = 0
                        elif "object" in prop_type:
                            data[req_prop] = None
                        else:
                            data[req_prop] = "N/A"

            coerced_dict = {}
            for k, v in data.items():
                if k in properties:
                    if k == "profiles" and isinstance(v, list):
                        filtered_profiles = []
                        from src.models.profile import Platform
                        valid_platforms = {p.value for p in Platform}

                        for item in v:
                            if isinstance(item, dict):
                                plat = item.get("platform")
                                url = item.get("url")
                                # Skip invalid platforms or profiles without urls
                                if plat in valid_platforms and url:
                                    filtered_profiles.append(item)
                        v = filtered_profiles

                    coerced_dict[k] = self._coerce_json_by_schema(v, properties[k], root_schema)
                else:
                    coerced_dict[k] = v
            return coerced_dict

        return data

    def _load_groq_key(self) -> str | None:
        import os
        if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"].strip():
            return os.environ["GROQ_API_KEY"].strip()

        # Search up the directory tree for .env
        curr = Path(__file__).resolve().parent
        for _ in range(6):
            env_path = curr / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() == "GROQ_API_KEY":
                                    val = v.strip().strip("'\"")
                                    if val:
                                        return val
                except Exception:
                    pass
            if curr.parent == curr:
                break
            curr = curr.parent
        return None

    def _extract_via_groq(self, text: str, candidate: ExtractedCandidate) -> ExtractedCandidate:
        groq_key = self._load_groq_key()
        if not groq_key:
            raise ValueError("GROQ_API_KEY not configured.")

        import json
        schema_dict = ExtractedCandidate.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2)

        system_prompt = (
            "You are an expert resume information extraction engine.\n\n"
            "Your task is to extract structured candidate information from the provided resume text.\n\n"
            "Rules:\n"
            "* Extract only information explicitly present in the resume.\n"
            "* Never hallucinate facts.\n"
            "* If a value is unknown, use null or an empty array depending on the schema.\n"
            "* Merge project titles, technology stack lines and bullet points into a single project object.\n"
            "* Never create duplicate projects.\n"
            "* Never treat technology stack lines (for example \"Java, MySQL\" or \"React, Node.js, MongoDB\") as project titles.\n"
            "* Preserve company names exactly.\n"
            "* Preserve dates exactly as written.\n"
            "* Preserve emails, phone numbers and URLs exactly.\n"
            "* Populate the skills list using technologies explicitly mentioned in the resume.\n"
            "* Do not invent work experience.\n"
            "* Do not infer certifications that are not written.\n"
            "* Return ONLY valid JSON.\n"
            "* Do not wrap the response in Markdown.\n"
            "* Do not include explanations.\n"
            "* Do not include comments.\n\n"
            "Most importantly, return the JSON conforming exactly to the following Pydantic JSON Schema:\n"
            f"{schema_json}\n\n"
            "Ensure that:\n"
            "* Every field must match the exact type defined by the application's ExtractedCandidate schema.\n"
            "* Do not return null for non-nullable fields.\n"
            "* If a field is a boolean, always return either true or false.\n"
            "* If a field is a string, always return a string or null only if the schema allows it.\n"
            "* If a field is an array, always return an array.\n"
            "* If a field is an object, always return an object matching the schema.\n"
            "* Do not change field types based on confidence or missing information.\n"
            "* Every required field is present.\n"
            "* Every field has the correct type.\n"
            "* Nested objects must exactly match the schema.\n"
            "* Arrays must contain objects matching the schema.\n"
            "* String fields must be strings, not arrays.\n"
            "* Object fields must be objects, not strings.\n"
            "* The output must be directly deserializable into ExtractedCandidate without any transformation.\n"
            "* If 'location' is defined as an object, return an object—not a string like 'Delhi, India'.\n"
            "* If 'projects[].description' is defined as a string, concatenate bullet points into a single newline-separated string instead of returning an array.\n"
            "* If 'projects[].company' is required, always provide it (use 'Personal Project' for personal projects)."
        )

        from openai import OpenAI, RateLimitError, InternalServerError, APITimeoutError, APIStatusError
        import time

        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )

        max_attempts = 2
        attempt = 0
        response = None

        while attempt < max_attempts:
            attempt += 1
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Resume Text:\n\n{text}"}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=30
                )
                break
            except (RateLimitError, InternalServerError, APITimeoutError) as e:
                if attempt >= max_attempts:
                    raise e
                print(f"Groq temporary failure ({e}), retrying once...")
                time.sleep(1)
            except APIStatusError as e:
                if e.status_code in (429, 503):
                    if attempt >= max_attempts:
                        raise e
                    print(f"Groq temporary failure ({e.status_code}), retrying once...")
                    time.sleep(1)
                else:
                    raise e

        if not response or not response.choices or not response.choices[0].message.content:
            raise ValueError("Groq returned an empty response.")

        json_str = response.choices[0].message.content.strip()

        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            profile_data = json.loads(json_str)
        except Exception as exc:
            print("Groq returned invalid JSON.")
            raise exc

        # Coerce/normalize parsed LLM response based on schema rules (Secondary safeguard)
        try:
            profile_data = self._coerce_json_by_schema(profile_data, schema_dict, schema_dict)
        except Exception as exc:
            print(f"JSON coercion failed: {exc}")

        profile_data["extraction_id"] = candidate.extraction_id
        profile_data["source_type"] = candidate.source_type
        profile_data["source_id"] = candidate.source_id
        profile_data["adapter_name"] = candidate.adapter_name
        profile_data["extracted_at"] = candidate.extracted_at
        profile_data["raw_text"] = candidate.raw_text

        try:
            validated = ExtractedCandidate.model_validate(profile_data)
            print("Groq extraction successful.")
            return validated
        except Exception as exc:
            print("Candidate validation failed.")
            raise exc

    def _extract(self, source: Union[Path, str, bytes]) -> ExtractedCandidate:
        """
        Full extraction pipeline: PDF → text → sections → fields.
        """
        source_id = self._build_source_id(source)
        candidate = self._new_candidate(source_id)

        # Step 1: PDF → raw text
        try:
            text = self._pdf_to_text(source)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                self.adapter_name, source_id, f"PDF parsing failed: {exc}"
            ) from exc

        if not text or not text.strip():
            candidate.add_warning(
                field="raw_text",
                message=(
                    "PDF appears empty or non-text (may be a scanned image). "
                    "OCR is not supported in this version."
                ),
            )
            return candidate

        candidate.raw_text = text

        groq_key = self._load_groq_key()
        if groq_key:
            print("GROQ_API_KEY detected.")
            print("Using Groq resume extraction.")
            try:
                groq_candidate = self._extract_via_groq(text, candidate)
                return groq_candidate
            except Exception as exc:
                print(f"Falling back to heuristic parser. Reason: {exc}")
                candidate.add_warning(
                    field="raw_text",
                    message=f"Groq LLM extraction failed: {exc}. Fell back to heuristic parser."
                )
        else:
            print("GROQ_API_KEY not configured.")
            print("Using heuristic parser.")

        from src.adapters.section_detector import SectionDetector
        detector = SectionDetector()
        detected_sections = detector.detect(text)

        # Step 2: extract each field group
        header_content = detected_sections.get("header", {}).get("content", "")
        header_lines = [ln.strip() for ln in header_content.splitlines() if ln.strip()]
        self._extract_name(candidate, header_lines)

        self._extract_emails(candidate, text)
        self._extract_phones(candidate, text)
        self._extract_links(candidate, text)

        self._extract_summary(candidate, detected_sections)
        self._extract_skills(candidate, detected_sections.get("skills", {}).get("content", ""))
        self._extract_experience(candidate, detected_sections.get("experience", {}).get("content", ""))
        self._extract_education(candidate, detected_sections.get("education", {}).get("content", ""))
        self._extract_projects(candidate, detected_sections.get("projects", {}).get("content", ""))

        return candidate

    # ── PDF text extraction ───────────────────────────────────────────────────

    def _pdf_to_text(self, source: Union[Path, str, bytes]) -> str:
        """
        Extract all text from a PDF using pdfplumber.

        Concatenates text from every page with a double newline between pages.
        pdfplumber is imported here (not at module level) so the module can be
        imported in test environments where pdfplumber may not be installed,
        as long as this method is mocked.

        Args:
            source: File path (Path/str) or raw bytes.

        Returns:
            Full extracted text as a single string.

        Raises:
            AdapterError: If the file cannot be opened or parsed.
        """
        try:
            import pdfplumber  # lazy import
        except ImportError as exc:
            raise AdapterError(
                self.adapter_name,
                "pdfplumber",
                "pdfplumber is not installed. Run: pip install pdfplumber",
            ) from exc

        file_obj = io.BytesIO(source) if isinstance(source, bytes) else source
        pages_text: list[str] = []

        try:
            with pdfplumber.open(file_obj) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
        except Exception as exc:
            raise AdapterError(
                self.adapter_name,
                str(source) if not isinstance(source, bytes) else "bytes",
                f"pdfplumber failed to open/read PDF: {exc}",
            ) from exc

        return "\n\n".join(pages_text)

    # ── Section detection ─────────────────────────────────────────────────────

    def _split_into_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """
        Legacy section splitter. Delegated to SectionDetector for backward compatibility.
        """
        from src.adapters.section_detector import SectionDetector
        detector = SectionDetector()
        detected = detector.detect("\n".join(lines))
        return {k: v["content"].splitlines() for k, v in detected.items()}

    def _detect_section_header(self, line: str) -> Optional[str]:
        """
        Legacy header detector. Delegated to SectionDetector for backward compatibility.
        """
        from src.adapters.section_detector import SectionDetector
        detector = SectionDetector()
        return detector._detect_heading(line, "")

    # ── Name extraction ───────────────────────────────────────────────────────

    def _extract_name(
        self, candidate: ExtractedCandidate, lines: list[str]
    ) -> None:
        """
        Extract candidate name from the top of the resume.

        Heuristic:
        1. Look at the first 12 non-empty lines.
        2. Skip lines containing contact signals (email, phone, URL, digits).
        3. Skip lines that are section headers.
        4. Skip lines containing known job-title keywords.
        5. The first remaining line that has 2–5 words with at least
           two starting in uppercase is likely the name.

        Sets: candidate.full_name, candidate.first_name, candidate.last_name,
        and candidate.middle_name (if applicable).
        """
        checked = 0
        for line in lines:
            if checked >= 12:
                break
            line = line.strip()
            if not line:
                continue
            checked += 1

            # Skip lines with obvious contact data
            if (
                _EMAIL_RE.search(line)
                or _PHONE_RE.search(line)
                or _URL_RE.search(line)
                or re.search(r"\d{4}", line)  # years → dates/zip codes
                or "|" in line  # contact separator: "Jane | jane@x.com | NYC"
            ):
                continue

            # Skip section headers
            if self._detect_section_header(line):
                continue

            # Skip lines with job-title keywords
            words_lower = set(line.lower().split())
            if words_lower & _TITLE_KEYWORDS:
                continue

            # Name heuristic: 2–5 words, most start with uppercase
            words = line.split()
            if not (2 <= len(words) <= 5):
                continue
            upper_count = sum(1 for w in words if w and w[0].isupper())
            if upper_count < max(2, len(words) - 1):
                continue

            # Reconstruct from possibly all-caps source
            name = " ".join(w.title() if w.isupper() else w for w in words)
            candidate.full_name = name
            candidate.first_name = words[0].title() if words[0].isupper() else words[0]
            candidate.last_name = words[-1].title() if words[-1].isupper() else words[-1]
            if len(words) == 3:
                candidate.middle_name = (
                    words[1].title() if words[1].isupper() else words[1]
                )
            return

        candidate.add_warning(
            field="full_name",
            message="Could not detect candidate name from the top of the resume.",
        )

    # ── Email extraction ──────────────────────────────────────────────────────

    def _extract_emails(
        self, candidate: ExtractedCandidate, text: str
    ) -> None:
        """
        Extract all email addresses from the full resume text.

        Uses the permissive email regex and deduplicates by lowercase address.
        The first unique email found is flagged as primary.
        """
        seen: set[str] = set()
        for match in _EMAIL_RE.finditer(text):
            raw_addr = match.group(0)
            try:
                email = Email(
                    address=raw_addr,
                    is_primary=len(candidate.emails) == 0,
                )
                if email.address in seen:
                    continue
                seen.add(email.address)
                candidate.emails.append(email)
            except (ValidationError, ValueError) as exc:
                candidate.add_warning(
                    field="email",
                    message=f"Could not parse email: {exc}",
                    raw=raw_addr,
                )

    # ── Phone extraction ──────────────────────────────────────────────────────

    def _extract_phones(
        self, candidate: ExtractedCandidate, text: str
    ) -> None:
        """
        Extract all phone numbers from the full resume text.

        Stores the raw matched string; the phone normaliser will parse and
        validate it later using the ``phonenumbers`` library.
        Deduplicates by raw string to avoid multiple matches on the same number.
        """
        seen: set[str] = set()
        for match in _PHONE_RE.finditer(text):
            raw = match.group(0).strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            candidate.phones.append(
                Phone(
                    raw=raw,
                    is_primary=len(candidate.phones) == 0,
                )
            )

    # ── Links / profile extraction ────────────────────────────────────────────

    def _extract_links(
        self, candidate: ExtractedCandidate, text: str
    ) -> None:
        """
        Extract all URLs and classify them into Profile objects.

        Uses platform-domain detection to distinguish GitHub from LinkedIn
        from personal portfolio sites. Deduplicates by normalised URL.
        """
        seen: set[str] = set()
        for match in _URL_RE.finditer(text):
            url = match.group(0).rstrip(".,;)'\"")
            url_norm = url.rstrip("/").lower()
            if url_norm in seen:
                continue
            seen.add(url_norm)
            platform = detect_platform(url)
            candidate.profiles.append(Profile(platform=platform, url=url))

    # ── Summary extraction ────────────────────────────────────────────────────

    def _extract_summary(
        self,
        candidate: ExtractedCandidate,
        detected_sections: dict[str, dict[str, Any]] | dict[str, list[str]],
        lines: Optional[list[str]] = None,
    ) -> None:
        """
        Extract the professional summary or objective.
        """
        # Handle dict[str, dict[str, Any]] (new) vs dict[str, list[str]] (legacy)
        if detected_sections and isinstance(next(iter(detected_sections.values())), dict):
            sections_dict = detected_sections  # type: ignore
            summary_content = sections_dict.get("summary", {}).get("content", "")
            if summary_content.strip():
                candidate.summary = summary_content.strip()
                return
            header_content = sections_dict.get("header", {}).get("content", "")
            header_lines = header_content.splitlines()
        else:
            sections_old = detected_sections  # type: ignore
            summary_lines = sections_old.get("summary", [])
            if summary_lines:
                text = "\n".join(ln for ln in summary_lines if ln.strip())
                if text.strip():
                    candidate.summary = text.strip()
                    return
            header_lines = sections_old.get("header", [])

        # Fallback: look for the first multi-word paragraph in the header
        paragraph: list[str] = []
        for ln in header_lines:
            stripped = ln.strip()
            if not stripped:
                if paragraph:
                    break
                continue
            if (
                _EMAIL_RE.search(stripped)
                or _PHONE_RE.search(stripped)
                or _URL_RE.search(stripped)
            ):
                continue
            if len(stripped.split()) >= 8:
                paragraph.append(stripped)

        if paragraph:
            candidate.summary = " ".join(paragraph)

    # ── Skills extraction ─────────────────────────────────────────────────────

    def _extract_skills(
        self,
        candidate: ExtractedCandidate,
        skills_content: dict[str, list[str]] | str,
    ) -> None:
        """
        Extract skills from the skills section.
        """
        if isinstance(skills_content, str):
            skill_lines = skills_content.splitlines() if skills_content else []
        else:
            skill_lines = skills_content.get("skills", [])

        if not skill_lines:
            return

        seen: set[str] = set()
        for line in skill_lines:
            line = line.strip()
            if not line:
                continue

            # Strip category labels: "Languages: Python, Go" → "Python, Go"
            if ":" in line:
                line = line.split(":", 1)[1]

            # Remove bullet markers
            line = _BULLET_PREFIX_RE.sub("", line)

            # Split on common skill delimiters
            raw_skills = re.split(r"[,;|•·▪◦○●]", line)
            for raw in raw_skills:
                name = raw.strip().strip('"').strip("'")
                # Filter noise: too short, too long, all numeric
                if not name or len(name) < 2 or len(name) > 60:
                    continue
                if name.isdigit():
                    continue
                name_lower = name.lower()
                if name_lower in seen:
                    continue
                seen.add(name_lower)
                candidate.skills.append(
                    Skill(
                        name=name,
                        source_context="resume:skills_section",
                    )
                )

    # ── Experience extraction ─────────────────────────────────────────────────

    def _extract_experience(
        self,
        candidate: ExtractedCandidate,
        experience_content: dict[str, list[str]] | str,
    ) -> None:
        """
        Extract work experience entries from the experience section.
        """
        if isinstance(experience_content, str):
            exp_lines = experience_content.splitlines() if experience_content else []
        else:
            exp_lines = experience_content.get("experience", [])

        if not exp_lines:
            return

        for block in self._split_experience_into_blocks(exp_lines):
            entry = self._parse_experience_block(block)
            if entry:
                candidate.experience.append(entry)

    def _split_experience_into_blocks(self, lines: list[str]) -> list[list[str]]:
        """
        Split experience lines into blocks using date ranges as anchors.

        Each block is guaranteed to have exactly one date range. Any lines before
        a date range (that are not bullet points and not part of a previous entry)
        are grouped as company/title headers.
        """
        # Find indices of all lines containing a date range
        date_indices = []
        for i, line in enumerate(lines):
            if _DATE_RANGE_RE.search(line):
                date_indices.append(i)

        if not date_indices:
            # Fallback to blank-line splitting if no date ranges are found
            return self._split_into_blocks(lines)

        blocks: list[list[str]] = []
        for idx, date_idx in enumerate(date_indices):
            # Determine the start of this experience entry.
            # It starts at the first non-bullet line before the date_idx
            # that is not already claimed by a previous entry.
            prev_boundary = date_indices[idx - 1] + 1 if idx > 0 else 0
            start_idx = date_idx
            while start_idx > prev_boundary:
                prev_line = lines[start_idx - 1].strip()
                if _BULLET_PREFIX_RE.match(prev_line) or not prev_line:
                    break
                start_idx -= 1

            # Determine the end of this experience entry.
            # It runs until the start_idx of the next entry, or the end of lines.
            if idx + 1 < len(date_indices):
                next_date_idx = date_indices[idx + 1]
                next_start_idx = next_date_idx
                while next_start_idx > date_idx + 1:
                    prev_line = lines[next_start_idx - 1].strip()
                    if _BULLET_PREFIX_RE.match(prev_line) or not prev_line:
                        break
                    next_start_idx -= 1
                end_idx = next_start_idx
            else:
                end_idx = len(lines)

            block = [l for l in lines[start_idx:end_idx] if l.strip()]
            if block:
                blocks.append(block)

        return blocks

    def _parse_experience_block(self, block: list[str]) -> Optional[Experience]:
        """
        Parse one blank-line-delimited block into an Experience.

        Algorithm:
        1. Find the line containing the date range (the temporal anchor).
        2. Lines before the date line → job title and company.
        3. Lines after the date line → description bullets.
        4. Handle pipe-separated format: "Title | Company | Date Range".
        5. Parse date range into start/end date objects.

        Returns None if the block has no date range (not a valid entry).
        """
        non_empty = [ln.strip() for ln in block if ln.strip()]
        if not non_empty:
            return None

        # Check for pipe-separated compact format: Title | Company | Date
        if "|" in non_empty[0]:
            parts = [p.strip() for p in non_empty[0].split("|")]
            date_part = next((p for p in parts if _DATE_RANGE_RE.search(p)), None)
            if date_part and len(parts) >= 2:
                other_parts = [p for p in parts if p != date_part]
                title = other_parts[0] if len(other_parts) >= 1 else "Unknown"
                company = other_parts[1] if len(other_parts) >= 2 else "Unknown"
                start_date, raw_start, end_date, raw_end, is_current = (
                    self._parse_date_range(date_part)
                )
                desc_lines = non_empty[1:]
                return self._build_experience(
                    title, company, start_date, raw_start,
                    end_date, raw_end, is_current, desc_lines,
                )

        # Standard multi-line format
        date_line_idx: Optional[int] = None
        for i, line in enumerate(non_empty):
            if _DATE_RANGE_RE.search(line):
                date_line_idx = i
                break

        if date_line_idx is None:
            return None  # No date range → not a parseable entry

        date_match = _DATE_RANGE_RE.search(non_empty[date_line_idx])
        raw_range = date_match.group(0) if date_match else ""
        start_date, raw_start, end_date, raw_end, is_current = (
            self._parse_date_range(raw_range)
        )

        # Text before the date line = title and company
        pre_date = non_empty[:date_line_idx]
        # Text after the date line = description
        post_date = non_empty[date_line_idx + 1:]

        title, company = self._infer_title_company(pre_date, non_empty[date_line_idx])

        return self._build_experience(
            title, company, start_date, raw_start,
            end_date, raw_end, is_current, post_date,
        )

    def _infer_title_company(
        self, pre_date: list[str], date_line: str
    ) -> tuple[str, str]:
        """
        Infer job title and company name from the lines before the date.

        Common patterns:
        - 0 pre-date lines: extract from date line remnant after removing dates
        - 1 pre-date line: could be "Title, Company" or just title or company
        - 2+ pre-date lines: first = title, second = company (most common)
        """
        if len(pre_date) >= 2:
            title = pre_date[0]
            company = pre_date[1]
            if "," in company:
                company = company.split(",", 1)[0].strip()
            return title, company

        if len(pre_date) == 1:
            line = pre_date[0]
            # Check if the date line itself contains a job title remnant
            remnant = _DATE_RANGE_RE.sub("", date_line).strip(" |–—-·")
            if remnant and len(remnant.split()) >= 1:
                # Layout:
                # Line 1: Company, Location
                # Line 2: Title Dec 2025 – Feb 2026
                title = remnant
                company = line
                if "," in company:
                    company = company.split(",", 1)[0].strip()
                return title, company

            # Fallback: single pre-date line containing both title & company
            at_match = re.search(r"\bat\b", line, re.IGNORECASE)
            if at_match:
                return line[: at_match.start()].strip(), line[at_match.end() :].strip()
            comma_parts = [p.strip() for p in line.split(",", 1)]
            if len(comma_parts) == 2:
                return comma_parts[0], comma_parts[1]
            return line, "Unknown"

        # No pre-date lines: extract from the date line itself
        remnant = _DATE_RANGE_RE.sub("", date_line).strip(" |–—-·")
        if remnant:
            parts = [p.strip() for p in remnant.split(",", 1)]
            if len(parts) == 2:
                return parts[0], parts[1]
            return remnant, "Unknown"

        return "Unknown", "Unknown"

    def _build_experience(
        self,
        title: str,
        company: str,
        start_date: Optional[date],
        raw_start: Optional[str],
        end_date: Optional[date],
        raw_end: Optional[str],
        is_current: bool,
        desc_lines: list[str],
    ) -> Experience:
        """Construct an Experience from parsed components."""
        responsibilities, achievements = self._classify_bullets(desc_lines)
        description = "\n".join(desc_lines).strip()

        return Experience(
            company=company or "Unknown",
            title=title or "Unknown",
            start_date=start_date,
            raw_start_date=raw_start,
            end_date=end_date,
            raw_end_date=raw_end,
            is_current=is_current,
            description=description or None,
            responsibilities=responsibilities,
            achievements=achievements,
        )

    # ── Education extraction ──────────────────────────────────────────────────

    def _extract_education(
        self,
        candidate: ExtractedCandidate,
        education_content: dict[str, list[str]] | str,
    ) -> None:
        """
        Extract education records from the education section.
        """
        if isinstance(education_content, str):
            edu_lines = education_content.splitlines() if education_content else []
        else:
            edu_lines = education_content.get("education", [])

        if not edu_lines:
            return

        for block in self._split_into_blocks(edu_lines):
            entry = self._parse_education_block(block)
            if entry:
                candidate.education.append(entry)

    def _parse_education_block(self, block: list[str]) -> Optional[Education]:
        """
        Parse one education block into an Education.

        Expected format (many variations):
          University of California, Berkeley
          Bachelor of Science in Computer Science    2015 – 2019
          GPA: 3.9/4.0
          Relevant Coursework: Distributed Systems, ML
        """
        non_empty = [ln.strip() for ln in block if ln.strip()]
        if not non_empty:
            return None

        institution = non_empty[0]
        # Skip blocks that look like experience entries (date range on first line
        # with title keywords)
        if _DATE_RANGE_RE.search(institution):
            return None

        degree = None
        degree_level = DegreeLevel.UNKNOWN
        field_of_study = None
        start_date = end_date = None
        raw_start = raw_end = None
        gpa: Optional[float] = None
        gpa_scale: float = 4.0
        courses: list[str] = []

        for line in non_empty[1:]:
            # Date range
            date_match = _DATE_RANGE_RE.search(line)
            if date_match:
                start_date, raw_start, end_date, raw_end, _ = self._parse_date_range(
                    date_match.group(0)
                )
                continue

            # GPA
            gpa_match = _GPA_RE.search(line)
            if gpa_match:
                gpa = float(gpa_match.group(1))
                if gpa_match.group(2):
                    gpa_scale = float(gpa_match.group(2))
                continue

            # Coursework
            if re.search(r"course|curriculum|coursework", line, re.IGNORECASE):
                # "Relevant Coursework: A, B, C"
                course_text = re.sub(r".*?coursework[:\s]*", "", line, flags=re.IGNORECASE)
                courses = [c.strip() for c in re.split(r"[,;]", course_text) if c.strip()]
                continue

            # Degree line
            detected_level = self._detect_degree_level(line)
            if detected_level != DegreeLevel.UNKNOWN:
                degree = line
                degree_level = detected_level
                # Try to extract field of study: "B.S. in Computer Science"
                in_match = re.search(r"\bin\b\s+(.+)", line, re.IGNORECASE)
                if in_match:
                    field_of_study = in_match.group(1).strip()
                continue

        # Validate: a school name should have at least 2 words or be ≥ 2 chars
        if len(institution.split()) < 1 or len(institution) < 2:
            return None

        try:
            return Education(
                institution=institution,
                degree=degree,
                degree_level=degree_level,
                field_of_study=field_of_study,
                start_date=start_date,
                raw_start_date=raw_start,
                end_date=end_date,
                raw_end_date=raw_end,
                gpa=gpa,
                gpa_scale=gpa_scale,
                courses=courses,
            )
        except (ValidationError, ValueError):
            return None

    # ── Projects extraction ───────────────────────────────────────────────────

    def _is_new_project_title(self, line: str) -> bool:
        line_strip = line.strip()
        if not line_strip:
            return False

        # Clean bullet prefix for classification check
        cleaned = _BULLET_PREFIX_RE.sub("", line_strip).strip()
        if not cleaned:
            return False

        # Negative Signals check:
        # 1. Starts with action verbs
        words = cleaned.split()
        if not words:
            return False
        first_word = words[0].lower().rstrip(",")
        action_verbs = {
            "built", "designed", "implemented", "optimized", "developed", "created",
            "worked", "managed", "offloaded", "led", "architected", "improved",
            "scaled", "investigated", "diagnosed", "engineered"
        }
        if first_word in action_verbs:
            return False

        # 2. Ends with a period
        if cleaned.endswith("."):
            return False

        # 3. Starts with lowercase letter
        if cleaned[0].islower():
            return False

        # 4. Too long (typically titles are short)
        if len(words) > 15:
            return False

        # Positive Signals check:
        # 1. Contains separators
        if any(sep in cleaned for sep in ["—", "–", " | ", "|", "~"]):
            return True

        # 2. Contains GitHub or Live or website links
        if any(kw in cleaned.lower() for kw in ["github", "live", "http", "www."]):
            return True

        # 3. Explicitly looks like title: short (< 8 words) and does not end with sentence punctuation
        if len(words) < 8 and not cleaned.endswith((".", ";", ",")):
            return True

        return False

    def _is_metadata_line(self, line: str) -> bool:
        line_strip = line.strip()
        if not line_strip:
            return False
        
        # Check explicit prefix
        if any(line_strip.lower().startswith(prefix) for prefix in ["tech stack:", "technologies:", "tech:", "stack:", "built with:"]):
            return True
        
        # Check if it consists purely of links, github, live, and separators
        cleaned = _BULLET_PREFIX_RE.sub("", line_strip).strip()
        cleaned_lower = cleaned.lower()
        
        # Replace symbols/separators with spaces to split
        s = re.sub(r"[|\-—–~/,;:]", " ", cleaned_lower)
        words = s.split()
        if not words:
            return False
        
        metadata_keywords = {"github", "live", "demo", "deployed", "website", "http", "https", "link", "links", "code", "repo", "repository", "preview", "live link", "website link"}
        
        is_pure_metadata = True
        for w in words:
            if not (w in metadata_keywords or w.startswith("http") or w.startswith("www") or w.isdigit()):
                is_pure_metadata = False
                break
        if is_pure_metadata:
            return True
            
        return False

    def _classify_line(self, line: str, current_project_exists: bool) -> str:
        line_strip = line.strip()
        if not line_strip:
            return "EMPTY"

        # 1. Check if it's metadata
        if self._is_metadata_line(line_strip):
            return "METADATA"

        # 2. Check if it's a new project
        if self._is_new_project_title(line_strip):
            return "NEW_PROJECT"

        # If no project exists yet, default to starting one so we don't drop lines
        if not current_project_exists:
            return "NEW_PROJECT"

        # 3. Check if it starts with a bullet point character
        if _BULLET_PREFIX_RE.match(line_strip):
            return "BULLET"

        # 4. Check if it starts with an action verb (indicates new bullet point)
        cleaned = _BULLET_PREFIX_RE.sub("", line_strip).strip()
        words = cleaned.split()
        if words:
            first_word = words[0].lower().rstrip(",")
            action_verbs = {
                "built", "designed", "implemented", "optimized", "developed", "created",
                "worked", "managed", "offloaded", "led", "architected", "improved",
                "scaled", "investigated", "diagnosed", "engineered"
            }
            if first_word in action_verbs:
                return "BULLET"

        # Default fallback: continuation line
        return "CONTINUATION"

    def _normalize_technology(self, tech: str) -> str:
        tech_lower = tech.lower()
        if tech_lower in ["node js", "nodejs"]:
            return "Node.js"
        if tech_lower in ["fast api", "fastapi"]:
            return "FastAPI"
        if tech_lower in ["mongo db", "mongodb"]:
            return "MongoDB"
        if tech_lower in ["c plus plus", "c++"]:
            return "C++"
        if tech_lower in ["rest api", "restapis", "restapi"]:
            return "REST APIs"
        return tech

    def _extract_techs_from_line(self, line: str) -> list[str]:
        # Check if line has explicit tech stack prefix:
        tech_prefix_match = re.match(
            r"(?:tech(?:nologies)?|stack|built with|tools?)[:\s]+(.+)",
            line,
            re.IGNORECASE,
        )
        raw_techs = []
        if tech_prefix_match:
            raw_techs = re.split(r"[,;|]", tech_prefix_match.group(1))
        else:
            # If it's a project title line, e.g. "MindTrack — AI-Powered Mental Wellness Platform — React.js, Node.js"
            if any(sep in line for sep in ["—", "–", " | ", "|"]):
                parts = re.split(r"—|–|\|", line)
                if len(parts) >= 2:
                    last_part = parts[-1].strip()
                    # If it has commas or is short, it's likely a tech list
                    if "," in last_part or len(last_part.split()) <= 5:
                        raw_techs = re.split(r"[,;]", last_part)

        cleaned_techs = []
        for t in raw_techs:
            t_clean = t.strip().rstrip(".,;)|")
            if not t_clean or len(t_clean) < 2 or len(t_clean) > 30:
                continue
            t_norm = self._normalize_technology(t_clean)
            if t_norm not in cleaned_techs:
                cleaned_techs.append(t_norm)

        # Scanner for known technologies in the text line (e.g. inside bullet points)
        known_tech_map = {
            "node.js": "Node.js", "node js": "Node.js", "nodejs": "Node.js",
            "fastapi": "FastAPI", "fast api": "FastAPI",
            "mongodb": "MongoDB", "mongo db": "MongoDB",
            "c++": "C++", "c plus plus": "C++",
            "rest api": "REST APIs", "restapis": "REST APIs", "restapi": "REST APIs",
            "python": "Python", "react": "React.js", "react.js": "React.js",
            "docker": "Docker", "kubernetes": "Kubernetes", "redis": "Redis",
            "express": "Express.js", "express.js": "Express.js", "go": "Go",
            "golang": "Go", "postgresql": "PostgreSQL", "postgres": "PostgreSQL"
        }

        line_lower = line.lower()
        for pattern, normalized in known_tech_map.items():
            if pattern in ["c++", "go"]:
                if pattern == "c++":
                    if "c++" in line_lower:
                        if normalized not in cleaned_techs:
                            cleaned_techs.append(normalized)
                elif pattern == "go":
                    if re.search(r"\bgo\b", line_lower):
                        if normalized not in cleaned_techs:
                            cleaned_techs.append(normalized)
            else:
                if re.search(r"\b" + re.escape(pattern) + r"\b", line_lower):
                    if normalized not in cleaned_techs:
                        cleaned_techs.append(normalized)

        return cleaned_techs

    def _extract_projects(
        self,
        candidate: ExtractedCandidate,
        projects_content: dict[str, list[str]] | str,
    ) -> None:
        """
        Extract project entries and store as Experience records using stateful parsing.
        """
        if isinstance(projects_content, str):
            proj_lines = projects_content.splitlines() if projects_content else []
        else:
            proj_lines = projects_content.get("projects", [])

        if not proj_lines:
            return

        projects_data = []
        current_project = None

        for line in proj_lines:
            line_strip = line.strip()
            if not line_strip:
                continue

            classification = self._classify_line(line_strip, current_project is not None)

            if classification == "NEW_PROJECT":
                title = _BULLET_PREFIX_RE.sub("", line_strip).strip()
                current_project = {
                    "title": title,
                    "description_lines": [],
                    "bullets": [],
                    "technologies": []
                }
                techs = self._extract_techs_from_line(title)
                current_project["technologies"].extend(techs)
                projects_data.append(current_project)

            elif classification == "METADATA":
                if current_project:
                    techs = self._extract_techs_from_line(line_strip)
                    current_project["technologies"].extend(techs)
                    current_project["description_lines"].append(line_strip)

            elif classification == "BULLET":
                if current_project:
                    bullet_text = _BULLET_PREFIX_RE.sub("", line_strip).strip()
                    current_project["bullets"].append(bullet_text)
                    techs = self._extract_techs_from_line(bullet_text)
                    current_project["technologies"].extend(techs)

            elif classification == "CONTINUATION":
                if current_project:
                    if current_project["bullets"]:
                        last_bullet = current_project["bullets"][-1]
                        current_project["bullets"][-1] = f"{last_bullet} {line_strip}"
                        techs = self._extract_techs_from_line(line_strip)
                        current_project["technologies"].extend(techs)
                    else:
                        current_project["description_lines"].append(line_strip)

        # Build Experience models for each project
        for p in projects_data:
            responsibilities, achievements = self._classify_bullets(p["bullets"])
            
            # Form clean description
            desc_parts = []
            if p["description_lines"]:
                desc_parts.extend(p["description_lines"])
            if p["bullets"]:
                desc_parts.extend(f"• {b}" for b in p["bullets"])
            description = "\n".join(desc_parts).strip() or None

            # De-duplicate technologies
            unique_techs = []
            for t in p["technologies"]:
                if t not in unique_techs:
                    unique_techs.append(t)

            candidate.projects.append(
                Experience(
                    company="Personal Project",
                    title=p["title"],
                    description=description,
                    responsibilities=responsibilities,
                    achievements=achievements,
                    technologies=unique_techs,
                    employment_type="project",
                )
            )

    # ── Shared parsing utilities ──────────────────────────────────────────────

    @staticmethod
    def _split_into_blocks(lines: list[str]) -> list[list[str]]:
        """
        Split a list of lines into blank-line-separated blocks.

        Consecutive non-empty lines form a block. A blank line (empty or
        whitespace-only) marks the end of a block.

        Args:
            lines: Lines from a named section.

        Returns:
            List of blocks, each block being a list of lines.
        """
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if line.strip():
                current.append(line)
            else:
                if current:
                    blocks.append(current)
                    current = []
        if current:
            blocks.append(current)
        return blocks

    def _parse_date_range(
        self, range_str: str
    ) -> tuple[Optional[date], Optional[str], Optional[date], Optional[str], bool]:
        """
        Parse a date range string into structured date components.

        Handles:
        - "Jan 2022 – Present"
        - "2019 - 2022"
        - "March 2018 to Present"
        - "2020 – Present"

        Args:
            range_str: Raw date range string extracted by _DATE_RANGE_RE.

        Returns:
            Tuple of (start_date, raw_start, end_date, raw_end, is_current).
        """
        # Split on dash/em-dash/en-dash variants
        parts = re.split(r"\s*[-–—]\s*", range_str.strip(), maxsplit=1)
        raw_start = parts[0].strip() if parts else None
        raw_end = parts[1].strip() if len(parts) > 1 else None

        is_current = bool(
            raw_end and raw_end.lower().strip() in {"present", "current", "now", "ongoing"}
        )

        start_date = self._parse_single_date(raw_start)
        end_date = None if is_current else self._parse_single_date(raw_end)

        return start_date, raw_start, end_date, raw_end, is_current

    @staticmethod
    def _parse_single_date(date_str: Optional[str]) -> Optional[date]:
        """
        Parse a single date string using python-dateutil.

        Uses a default of Jan 1, 2000 so year-only strings ("2019") produce
        a valid date(2019, 1, 1) rather than raising.

        Returns None if parsing fails, rather than raising.
        """
        if not date_str or date_str.lower() in {"present", "current", "now", "ongoing"}:
            return None
        try:
            from dateutil import parser as dateutil_parser
            from datetime import datetime as dt

            return dateutil_parser.parse(
                date_str, default=dt(2000, 1, 1, 0, 0)
            ).date()
        except Exception:
            return None

    def _detect_degree_level(self, text: str) -> DegreeLevel:
        """
        Detect the degree level from a line of text.

        Checks against the _DEGREE_MAP in specificity order (doctorate first)
        to avoid "Bachelor of Arts in Master Planning" matching as MASTER.

        Args:
            text: One line from the education block (e.g., "B.S. Computer Science").

        Returns:
            Matching DegreeLevel or DegreeLevel.UNKNOWN.
        """
        text_lower = f" {text.lower()} "
        for level, keywords in _DEGREE_MAP:
            for kw in keywords:
                if kw in text_lower:
                    return level
        return DegreeLevel.UNKNOWN

    @staticmethod
    def _classify_bullets(
        lines: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Separate description lines into responsibilities and achievements.

        Heuristic: a line is an achievement if it contains quantified outcomes
        (percentages, multipliers, dollar amounts, headcount figures).
        Everything else is a responsibility.

        Args:
            lines: Description/bullet lines from an experience block.

        Returns:
            (responsibilities, achievements) — two separate lists.
        """
        responsibilities: list[str] = []
        achievements: list[str] = []

        _ACHIEVEMENT_RE = re.compile(
            r"\d+\s*%"         # percentages: "40%"
            r"|\d+x"           # multipliers: "3x faster"
            r"|\$[\d,]+"       # dollar amounts
            r"|\d+\s*(?:million|billion|thousand|k)\b"  # large numbers
            r"|\d+\s*(?:engineers?|developers?|members?|users?)",  # headcount
            re.IGNORECASE,
        )

        for line in lines:
            clean = _BULLET_PREFIX_RE.sub("", line.strip())
            if not clean:
                continue
            if _ACHIEVEMENT_RE.search(clean):
                achievements.append(clean)
            else:
                responsibilities.append(clean)

        return responsibilities, achievements

    # ── Source ID ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_source_id(source: Union[Path, str, bytes]) -> str:
        """
        Build a deterministic, human-readable source_id.

        File path sources use the filename for readability.
        Bytes sources use a content hash for deduplication.
        """
        if isinstance(source, bytes):
            digest = hashlib.sha256(source).hexdigest()[:12]
            return f"resume:bytes:{digest}"
        return f"resume:{Path(source).name}"
