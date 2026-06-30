from src.models.education import DegreeLevel, Education
from src.models.email import Email
from src.models.experience import Experience
from src.models.extracted_candidate import ExtractedCandidate
from src.models.location import Location
from src.models.phone import Phone
from src.models.provenance import SourceType
from src.models.skill import Skill
from src.normalizers import CandidateNormalizer


def test_candidate_normalizer_returns_normalized_candidate():
    extracted = ExtractedCandidate(
        source_type=SourceType.MANUAL,
        source_id="manual:test",
        adapter_name="TestAdapter",
        full_name="jane DOE",
        emails=[Email(address="JANE@EXAMPLE.COM")],
        phones=[Phone(raw="(415) 555-2671")],
        location=Location(raw="San Francisco, CA, US"),
        skills=[Skill(name="js"), Skill(name="JavaScript")],
        experience=[
            Experience(
                company="Acme Inc.",
                title="Engineer",
                raw_start_date="Jan 2020",
                raw_end_date="Present",
            )
        ],
        education=[Education(institution="State University", degree="Bachelor of Science")],
    )

    normalized = CandidateNormalizer().normalize(extracted)

    assert normalized.first_name == "Jane"
    assert normalized.last_name == "Doe"
    assert normalized.emails[0].address == "jane@example.com"
    assert normalized.phones[0].normalized == "+14155552671"
    assert normalized.location.city == "San Francisco"
    assert normalized.location.state_code == "CA"
    assert [s.name for s in normalized.skills] == ["JavaScript"]
    assert normalized.experience[0].company == "Acme"
    assert normalized.experience[0].start_date.year == 2020
    assert normalized.education[0].degree_level == DegreeLevel.BACHELOR
    assert normalized.normalization_logs