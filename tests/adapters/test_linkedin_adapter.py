from src.adapters.linkedin_adapter import LinkedInAdapter
from src.models.provenance import SourceType
from src.adapters.linkedin_adapter import LinkedInAdapter
from src.models.provenance import SourceType
 

def test_linkedin_adapter_extracts_exported_profile():
    source = {
        "full_name": "Jane Doe",
        "headline": "Senior Software Engineer",
        "summary": "Backend engineer focused on hiring platforms.",
        "profile_url": "https://linkedin.com/in/janedoe",
        "experience": [
            {
                "company": "Acme Inc.",
                "title": "Senior Engineer",
                "start_date": "Jan 2020",
                "end_date": "Present",
                "description": "Built APIs.",
            }
        ],
        "education": [
            {
                "school": "State University",
                "degree": "Bachelor of Science",
                "fieldOfStudy": "Computer Science",
                "endDate": "2018",
            }
        ],
        "skills": ["Python", {"name": "Leadership"}],
    }

    candidate = LinkedInAdapter().extract(source)

    assert candidate.source_type == SourceType.LINKEDIN
    assert candidate.full_name == "Jane Doe"
    assert candidate.summary.startswith("Backend engineer")
    assert candidate.metadata["headline"] == "Senior Software Engineer"
    assert candidate.experience[0].company == "Acme Inc."
    assert candidate.education[0].institution == "State University"
    assert {s.name for s in candidate.skills} == {"Python", "Leadership"}


def test_linkedin_adapter_accepts_replaceable_parser():
    class Parser:
        def parse(self, source):
            return {
                "full_name": "Parser Person",
                "headline": "Parsed headline",
                "summary": "Parsed summary",
                "experience": [],
                "education": [],
                "skills": ["Go"],
            }

    candidate = LinkedInAdapter(parser=Parser()).extract({"anything": True})

    assert candidate.full_name == "Parser Person"
    assert candidate.skills[0].name == "Go"