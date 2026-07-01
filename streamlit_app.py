import json
import csv
from io import StringIO, BytesIO
from typing import Any, Optional

import streamlit as st

from src.models.extracted_candidate import ExtractedCandidate
from src.models.normalized_candidate import NormalizedCandidate
from src.models.canonical_candidate import CanonicalCandidate
from src.adapters.resume_adapter import ResumeAdapter
from src.adapters.csv_adapter import CSVAdapter
from src.adapters.ats_json_adapter import ATSJsonAdapter, GREENHOUSE_FIELD_MAPPING
from src.adapters.github_adapter import GitHubAdapter
from src.normalizers.candidate_normalizer import CandidateNormalizer
from src.engines.merge_engine import MergeEngine
from src.engines.conflict_resolver import ConflictResolver
from src.engines.confidence_engine import ConfidenceEngine
from src.projection.engine import ProjectionEngine, ProjectionConfig
from src.projection.schema_validator import JSONSchemaValidator


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for tests and app logic
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_text(text: str, fallback: dict) -> dict:
    """Parse JSON text and return a dict, falling back to a default if empty or invalid."""
    if not text or not text.strip():
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def parse_csv_upload(csv_upload) -> list[dict]:
    """Parse uploaded CSV file and return a list of dictionaries representing rows."""
    if csv_upload is None:
        return []
    try:
        content = csv_upload.getvalue().decode("utf-8")
        reader = csv.DictReader(StringIO(content))
        return [row for row in reader if any(row.values())]
    except Exception as exc:
        st.error(f"Error parsing CSV file: {exc}")
        return []


def parse_json_upload(ats_upload) -> dict:
    """Parse uploaded ATS JSON file and return a dictionary."""
    if ats_upload is None:
        return {}
    try:
        content = ats_upload.getvalue().decode("utf-8")
        return json.loads(content)
    except Exception as exc:
        st.error(f"Error parsing ATS JSON file: {exc}")
        return {}


def merge_sources(normalized_candidates: list[NormalizedCandidate]) -> CanonicalCandidate:
    """Run the MergeEngine, ConflictResolver, and ConfidenceEngine sequentially."""
    if not normalized_candidates:
        raise ValueError("Cannot merge an empty list of normalized candidates.")
    
    merge_eng = MergeEngine()
    resolver = ConflictResolver()
    confidence_eng = ConfidenceEngine()

    canonical = merge_eng.merge(normalized_candidates)
    resolved = resolver.resolve(canonical)
    scored = confidence_eng.score(resolved)
    
    return scored


def generate_output(canonical: CanonicalCandidate, config: dict, schema: Optional[dict] = None) -> dict:
    """Project the canonical profile."""
    projection_eng = ProjectionEngine()
    projected = projection_eng.project(canonical, config)
    if schema:
        validator = JSONSchemaValidator()
        validator.validate_or_raise(projected, schema)
    return projected


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration and schema
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "fields": {
        "candidate_id": "candidate_id",
        "display_name": "display_name",
        "primary_email": "primary_email",
        "primary_phone": "primary_phone",
        "summary": "summary.value",
        "location": {
            "path": "location.value",
            "flatten": True
        },
        "skills": {
            "path": "skills",
            "array": {
                "fields": {
                    "name": "name",
                    "confidence": "confidence"
                }
            }
        },
        "experience": {
            "path": "experience",
            "array": {
                "fields": {
                    "company": "company",
                    "title": "title",
                    "duration_months": "duration_months",
                    "is_current": "is_current"
                }
            }
        },
        "projects": {
            "path": "projects",
            "array": {
                "fields": {
                    "company": "company",
                    "title": "title",
                    "description": "description",
                    "is_current": "is_current"
                }
            }
        },
        "education": {
            "path": "education",
            "array": {
                "fields": {
                    "institution": "institution",
                    "degree": "degree",
                    "degree_level": "degree_level",
                    "gpa": "gpa"
                }
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Candidate Data Transformation Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚡ Candidate Data Transformation Engine")
    st.markdown(
        """
        Extract, normalize, merge, and project candidate profiles from unstructured resumes (PDF),
        structured spreadsheets (CSV), ATS exports (JSON), and social/professional developer portfolios.
        """
    )

    # Sidebar for Configuration
    st.sidebar.header("🔧 Projection Configuration")
    
    config_text = st.sidebar.text_area(
        "Config JSON",
        value=json.dumps(DEFAULT_CONFIG, indent=2),
        height=400,
        help="Define the runtime projection rules to map Canonical Candidate to client JSON."
    )
    config_json = parse_json_text(config_text, DEFAULT_CONFIG)

    # Main Grid Layout for Inputs
    st.header("📥 Input Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Unstructured Resume")
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        
        st.subheader("📊 Structured Spreadsheet")
        uploaded_csv = st.file_uploader("Upload Candidates (CSV)", type=["csv"])
        
        st.subheader("⚙️ ATS Profile JSON")
        uploaded_ats = st.file_uploader("Upload ATS Candidate (JSON)", type=["json"])
        
    with col2:
        st.subheader("🌐 Developer Portfolios")
        github_url = st.text_input("GitHub Username or URL", placeholder="e.g. janedoe or https://github.com/janedoe")
        
        st.subheader("🔑 API Secrets (Optional)")
        github_token = st.text_input("GitHub Personal Access Token", type="password", help="To avoid rate limits during repo crawling")

    # State Management for Pipeline
    if "final_output" not in st.session_state:
        st.session_state.final_output = None

    # Pipeline Actions Area
    st.header("⚙️ Pipeline Execution")
    
    generate_btn = st.button("🚀 Generate Output", use_container_width=True, type="primary")

    if generate_btn:
        has_input = (
            uploaded_resume is not None or
            uploaded_csv is not None or
            uploaded_ats is not None or
            bool(github_url.strip())
        )
        if not has_input:
            st.warning("Please upload or enter at least one candidate data source.")
            st.session_state.final_output = None
        else:
            extracted_list = []
            
            # 1. Extract
            with st.spinner("Extracting from input sources..."):
                # Resume
                if uploaded_resume:
                    try:
                        import time
                        start_time = time.time()
                        resume_adapter = ResumeAdapter()
                        res_extracted = resume_adapter.extract(uploaded_resume.getvalue())
                        elapsed_time = time.time() - start_time
                        extracted_list.append(res_extracted)
                        st.success(f"Resume extracted successfully in {elapsed_time:.2f} seconds!")
                    except Exception as exc:
                        st.error(f"Resume extraction failed: {exc}")

                # CSV
                if uploaded_csv:
                    csv_rows = parse_csv_upload(uploaded_csv)
                    csv_adapter = CSVAdapter()
                    try:
                        grouped_candidates = csv_adapter.extract_rows(csv_rows)
                        extracted_list.extend(grouped_candidates)
                        if grouped_candidates:
                            st.success(f"Extracted and grouped {len(grouped_candidates)} candidates from {len(csv_rows)} rows!")
                    except Exception as exc:
                        st.error(f"CSV extraction failed: {exc}")

                # ATS JSON
                if uploaded_ats:
                    ats_dict = parse_json_upload(uploaded_ats)
                    if ats_dict:
                        try:
                            ats_adapter = ATSJsonAdapter(field_mapping=GREENHOUSE_FIELD_MAPPING, source_label="greenhouse")
                            ats_extracted = ats_adapter.extract(ats_dict)
                            extracted_list.append(ats_extracted)
                            st.success("ATS JSON extracted successfully!")
                        except Exception as exc:
                            st.error(f"ATS JSON extraction failed: {exc}")

                # GitHub URL
                if github_url:
                    try:
                        github_adapter = GitHubAdapter(token=github_token if github_token else None)
                        gh_extracted = github_adapter.extract(github_url)
                        extracted_list.append(gh_extracted)
                        
                        has_github_api_warning = any(w.field == "github_api" for w in gh_extracted.warnings)
                        if has_github_api_warning:
                            st.warning("GitHub profile imported successfully with limited information. Some data could not be retrieved because the GitHub API rate limit was reached or authentication was unavailable.")
                            if not github_token:
                                st.info("Connect a GitHub Personal Access Token to increase API limits and enable more complete profile extraction.")
                        else:
                            st.success("GitHub profile extracted successfully!")
                    except Exception as exc:
                        st.error(f"GitHub extraction failed: {exc}")

            if not extracted_list:
                st.error("No candidate data could be extracted from the provided sources.")
                st.session_state.final_output = None
            else:
                # 2. Normalize
                with st.spinner("Normalizing candidate fields..."):
                    normalizer = CandidateNormalizer()
                    normalized_list = []
                    for ext in extracted_list:
                        try:
                            norm = normalizer.normalize(ext)
                            normalized_list.append(norm)
                        except Exception as exc:
                            st.error(f"Normalization failed for {ext.source_id}: {exc}")
                
                if not normalized_list:
                    st.error("Normalization failed for all candidates.")
                    st.session_state.final_output = None
                else:
                    # 3. Merge
                    canonical = None
                    with st.spinner("Merging profiles & computing confidence scores..."):
                        try:
                            canonical = merge_sources(normalized_list)
                            st.success("Merged profiles successfully!")
                        except Exception as exc:
                            st.error(f"Merging failed: {exc}")

                    if canonical is None:
                        st.session_state.final_output = None
                    else:
                        # 4. Project
                        with st.spinner("Projecting canonical profile..."):
                            try:
                                from src.projection.engine import ProjectionConfig
                                cfg_obj = ProjectionConfig.from_runtime(config_json)
                                for warning in cfg_obj.warnings:
                                    st.warning(warning)

                                projected = generate_output(
                                    canonical,
                                    cfg_obj
                                )
                                st.session_state.final_output = projected
                                st.success("Generated projected JSON successfully!")
                            except Exception as exc:
                                from src.projection import ProjectionError
                                if isinstance(exc, ProjectionError):
                                    st.error(f"Projection failed: {exc}")
                                    st.warning("Below is the data that was successfully projected before the failure:")
                                    st.json(exc.partial_data)
                                else:
                                    st.error(f"Projection failed: {exc}")

    # Results Visualizer
    st.header("📊 Final Output JSON")
    if st.session_state.final_output is not None:
        st.json(st.session_state.final_output)
    else:
        st.info("No projected output available. Please select input data, update the configuration if needed, and click 'Generate Output'.")


if __name__ == "__main__":
    main()

