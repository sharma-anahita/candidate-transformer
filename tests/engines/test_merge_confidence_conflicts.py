from src.engines import ConfidenceEngine, ConflictResolver, MergeEngine
from src.models.email import Email
from src.models.experience import Experience
from src.models.normalized_candidate import NormalizedCandidate
from src.models.phone import Phone
from src.models.provenance import SourceType
from src.models.skill import Skill


def candidate(source_type, source_id, adapter_name, **kwargs):
    return NormalizedCandidate(
        extraction_id=kwargs.pop("extraction_id", None) or __import__("uuid").uuid4(),
        source_type=source_type,
        source_id=source_id,
        adapter_name=adapter_name,
        **kwargs,
    )


def test_merge_deduplicates_lists_and_preserves_scalar_conflicts():
    a = candidate(
        SourceType.RESUME,
        "resume:1",
        "ResumeAdapter",
        first_name="Jane",
        emails=[Email(address="jane@example.com")],
        phones=[Phone(raw="4155552671", normalized="+14155552671", is_valid=True)],
        skills=[Skill(name="Python")],
        experience=[Experience(company="Acme", title="Engineer", raw_start_date="2020")],
    )
    b = candidate(
        SourceType.LINKEDIN,
        "linkedin:1",
        "LinkedInAdapter",
        first_name="Janet",
        emails=[Email(address="JANE@example.com")],
        phones=[Phone(raw="+1 415 555 2671", normalized="+14155552671", is_valid=True)],
        skills=[Skill(name="Python")],
        experience=[Experience(company="Acme Corporation", title="Engineer", raw_start_date="2020")],
    )

    merged = MergeEngine().merge([a, b])

    assert len(merged.emails) == 1
    assert len(merged.phones) == 1
    assert len(merged.skills) == 1
    assert merged.first_name is not None
    assert merged.first_name.conflicts


def test_confidence_engine_scores_fields_and_overall():
    c = candidate(
        SourceType.ATS_JSON,
        "ats:1",
        "ATSJsonAdapter",
        first_name="Jane",
        last_name="Doe",
        emails=[Email(address="jane@example.com")],
        skills=[Skill(name="Python")],
    )

    merged = MergeEngine().merge([c])
    scored = ConfidenceEngine().score(merged)

    assert scored.field_confidences["emails"] > 0
    assert scored.overall_confidence > 0


def test_conflict_resolver_prefers_structured_source():
    resume = candidate(SourceType.RESUME, "resume:1", "ResumeAdapter", first_name="Janet")
    ats = candidate(SourceType.ATS_JSON, "ats:1", "ATSJsonAdapter", first_name="Jane")

    merged = MergeEngine().merge([resume, ats])
    scored = ConfidenceEngine().score(merged)
    resolved = ConflictResolver().resolve(scored)

    assert resolved.first_name.value == "Jane"


def test_confidence_engine_github_skill_source_weight_boost():
    # Test Resume source weight is 0.80
    resume_candidate = candidate(
        SourceType.RESUME,
        "resume:1",
        "ResumeAdapter",
        skills=[Skill(name="Python")]
    )
    merged_resume = MergeEngine().merge([resume_candidate])
    scored_resume = ConfidenceEngine().score(merged_resume)
    assert scored_resume.skills[0].confidence == 0.56

    # Test GitHub skill source weight boost to 0.95
    github_candidate = candidate(
        SourceType.GITHUB,
        "github:1",
        "GitHubAdapter",
        skills=[Skill(name="Python")]
    )
    merged_github = MergeEngine().merge([github_candidate])
    scored_github = ConfidenceEngine().score(merged_github)
    assert scored_github.skills[0].confidence == 0.931


def test_confidence_engine_github_skill_occurrences_boost():
    github_candidate = candidate(
        SourceType.GITHUB,
        "github:1",
        "GitHubAdapter",
        skills=[Skill(name="Python", github_occurrence_count=2)]
    )
    merged = MergeEngine().merge([github_candidate])
    scored = ConfidenceEngine().score(merged)
    assert scored.skills[0].confidence == 0.981