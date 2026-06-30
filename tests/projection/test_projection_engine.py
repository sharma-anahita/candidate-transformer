from src.models.canonical_candidate import CanonicalCandidate, CanonicalSkill
from src.models.provenance import ConfidenceField


def test_projection_renames_flattens_maps_arrays_and_hides_metadata():
    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Jane", confidence=0.9, provenance=[]),
        last_name=ConfidenceField(value="Doe", confidence=0.9, provenance=[]),
        skills=[
            CanonicalSkill(name="Python", confidence=0.82, provenance=[]),
            CanonicalSkill(name="FastAPI", confidence=0.78, provenance=[]),
        ],
        overall_confidence=0.8,
    )

    config = {
        "hide_confidence": True,
        "hide_provenance": True,
        "fields": {
            "firstName": "first_name.value",
            "lastName": "last_name.value",
            "missingField": {"path": "does.not.exist", "missing": ""},
            "skills": {
                "path": "skills",
                "array": {
                    "fields": {
                        "label": "name",
                        "confidence": "confidence",
                        "provenance": "provenance",
                    }
                },
            },
        },
    }

    projected = __import__("src.projection", fromlist=["ProjectionEngine"]).ProjectionEngine().project(candidate, config)

    assert projected["firstName"] == "Jane"
    assert projected["lastName"] == "Doe"
    assert projected["missingField"] == ""
    assert projected["skills"] == [{"label": "Python"}, {"label": "FastAPI"}]
    assert "overall_confidence" not in projected


def test_projection_flattens_nested_object():
    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Jane", confidence=0.9, provenance=[]),
    )

    projected = __import__("src.projection", fromlist=["ProjectionEngine"]).ProjectionEngine().project(
        candidate,
        {
            "fields": {
                "name": {"path": "first_name", "flatten": True, "prefix": "first_"}
            }
        },
    )

    assert projected["name"]["first_value"] == "Jane"
    assert projected["name"]["first_confidence"] == 0.9


def test_projection_list_descriptors_and_on_missing():
    from src.projection import ProjectionEngine, ProjectionConfig
    import pytest

    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Anahita", confidence=0.9, provenance=[]),
        last_name=ConfidenceField(value="Sharma", confidence=0.9, provenance=[]),
        skills=[
            CanonicalSkill(name="js", confidence=0.8, provenance=[]),
        ]
    )

    config = {
        "on_missing": "null",
        "fields": [
            {
                "path": "candidate_name",
                "from": "display_name",
                "required": False,
                "type": "string"
            },
            {
                "path": "skills",
                "from": "skills",
                "type": "array",
                "array": {
                    "fields": [
                        {
                            "path": "skill_name",
                            "from": "name",
                            "normalize": "canonical"
                        }
                    ]
                }
            },
            {
                "path": "missing_omit",
                "from": "missing_field",
                "required": False
            }
        ]
    }

    projected = ProjectionEngine().project(candidate, config)
    assert projected["candidate_name"] == "Anahita Sharma"
    assert projected["skills"] == [{"skill_name": "JavaScript"}]
    assert projected["missing_omit"] is None

    config["on_missing"] = "omit"
    projected_omit = ProjectionEngine().project(candidate, config)
    assert "missing_omit" not in projected_omit

    config["on_missing"] = "error"
    with pytest.raises(ValueError, match="could not be resolved"):
        ProjectionEngine().project(candidate, config)

    config["on_missing"] = "null"
    config["fields"][0]["required"] = True
    config["fields"][0]["from"] = "nonexistent_field"
    with pytest.raises(ValueError, match="could not be resolved"):
        ProjectionEngine().project(candidate, config)


def test_projection_error_contains_partial_data():
    from src.projection import ProjectionEngine, ProjectionError
    import pytest

    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Anahita", confidence=0.9, provenance=[]),
        last_name=ConfidenceField(value="Sharma", confidence=0.9, provenance=[]),
    )

    config = {
        "on_missing": "null",
        "fields": [
            {
                "path": "candidate_name",
                "from": "display_name",
                "required": False,
                "type": "string"
            },
            {
                "path": "missing_required_1",
                "from": "nonexistent_field_1",
                "required": True
            },
            {
                "path": "another_good_field",
                "from": "first_name.value",
                "required": False
            },
            {
                "path": "missing_required_2",
                "from": "nonexistent_field_2",
                "required": True
            }
        ]
    }

    with pytest.raises(ProjectionError) as exc_info:
        ProjectionEngine().project(candidate, config)

    assert exc_info.value.partial_data == {
        "candidate_name": "Anahita Sharma",
        "another_good_field": "Anahita"
    }
    err_msg = str(exc_info.value)
    assert "missing_required_1" in err_msg
    assert "missing_required_2" in err_msg


def test_projection_array_recursive():
    from src.projection import ProjectionEngine
    
    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Rahul", confidence=0.9, provenance=[]),
        last_name=ConfidenceField(value="Sharma", confidence=0.9, provenance=[]),
        skills=[
            CanonicalSkill(name="Python", confidence=0.95, provenance=[]),
            CanonicalSkill(name="Java", confidence=0.87, provenance=[]),
        ]
    )
    
    config = {
        "on_missing": "null",
        "fields": [
            {
                "path": "skills",
                "from": "skills",
                "fields": [
                    {
                        "path": "name",
                        "from": "name"
                    },
                    {
                        "path": "confidence",
                        "from": "confidence"
                    }
                ]
            }
        ]
    }
    
    projected = ProjectionEngine().project(candidate, config)
    assert projected["skills"] == [
        {"name": "Python", "confidence": 0.95},
        {"name": "Java", "confidence": 0.87}
    ]


def test_projection_metadata_injection():
    from src.projection import ProjectionEngine
    from src.models.provenance import Provenance, SourceType, ExtractionMethod
    
    prov1 = Provenance(
        source_type=SourceType.MANUAL,
        adapter_name="manual",
        method=ExtractionMethod.MANUAL,
        source_id="1",
        raw_value="val",
        confidence=1.0
    )
    prov2 = Provenance(
        source_type=SourceType.MANUAL,
        adapter_name="manual",
        method=ExtractionMethod.MANUAL,
        source_id="2",
        raw_value="val",
        confidence=1.0
    )

    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Rahul", confidence=0.9, provenance=[prov1]),
        last_name=ConfidenceField(value="Sharma", confidence=0.9, provenance=[prov1]),
        overall_confidence=0.9,
        provenance={"first_name": [prov1], "last_name": [prov1]},
        skills=[
            CanonicalSkill(name="Python", confidence=0.95, provenance=[prov2]),
        ]
    )
    
    config = {
        "on_missing": "null",
        "include_confidence": True,
        "include_provenance": True,
        "fields": [
            {
                "path": "candidate_name",
                "from": "display_name"
            },
            {
                "path": "skills",
                "from": "skills",
                "fields": [
                    {
                        "path": "name",
                        "from": "name"
                    }
                ]
            }
        ]
    }
    
    projected = ProjectionEngine().project(candidate, config)
    assert projected["confidence"] == 0.9
    assert isinstance(projected["provenance"], dict)
    assert len(projected["provenance"]) == 2
    assert projected["provenance"]["first_name"][0]["source_id"] == "1"
    assert len(projected["skills"]) == 1
    skill = projected["skills"][0]
    assert skill["name"] == "Python"
    assert skill["confidence"] == 0.95
    assert skill["provenance"][0]["source_id"] == "2"


def test_projection_from_key_fallback():
    from src.projection import ProjectionEngine
    
    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Anahita", confidence=0.9, provenance=[]),
        last_name=ConfidenceField(value="Sharma", confidence=0.9, provenance=[]),
    )
    
    config = {
        "on_missing": "null",
        "fields": [
            {
                "path": "display_name"
            }
        ]
    }
    
    projected = ProjectionEngine().project(candidate, config)
    assert projected["display_name"] == "Anahita Sharma"


def test_projection_confidence_range_scaling():
    from src.projection import ProjectionEngine
    
    candidate = CanonicalCandidate(
        first_name=ConfidenceField(value="Rahul", confidence=0.84, provenance=[]),
        last_name=ConfidenceField(value="Sharma", confidence=0.84, provenance=[]),
        overall_confidence=0.84,
        skills=[
            CanonicalSkill(name="Python", confidence=0.95, provenance=[]),
        ]
    )
    
    # Range 0 to 100
    config_100 = {
        "on_missing": "null",
        "confidence_range": {"min": 0, "max": 100},
        "fields": [
            {"path": "candidate_name", "from": "display_name"},
            {"path": "confidence", "from": "confidence"},
            {
                "path": "skills",
                "from": "skills",
                "fields": [
                    {"path": "name", "from": "name"},
                    {"path": "confidence", "from": "confidence"}
                ]
            }
        ]
    }
    projected_100 = ProjectionEngine().project(candidate, config_100)
    assert projected_100["confidence"] == 84.0
    assert projected_100["skills"][0]["confidence"] == 95.0

    # Range 1 to 10
    config_10 = {
        "on_missing": "null",
        "confidence_range": {"min": 1, "max": 10},
        "fields": [
            {"path": "confidence", "from": "confidence"},
            {
                "path": "skills",
                "from": "skills",
                "fields": [
                    {"path": "confidence", "from": "confidence"}
                ]
            }
        ]
    }
    projected_10 = ProjectionEngine().project(candidate, config_10)
    # min + confidence * (max - min) = 1 + 0.84 * (10 - 1) = 1 + 7.56 = 8.56
    assert projected_10["confidence"] == 8.56
    # 1 + 0.95 * 9 = 9.55
    assert projected_10["skills"][0]["confidence"] == 9.55


def test_projection_confidence_range_validation():
    from src.projection import ProjectionEngine
    from src.projection.engine import ProjectionConfig
    
    candidate = CanonicalCandidate(
        overall_confidence=0.84
    )
    
    # Invalid range: min >= max
    config_invalid = {
        "on_missing": "null",
        "confidence_range": {"min": 100, "max": 50},
        "fields": [
            {"path": "confidence", "from": "confidence"}
        ]
    }
    
    cfg = ProjectionConfig.from_runtime(config_invalid)
    assert len(cfg.warnings) == 1
    assert "Invalid confidence range" in cfg.warnings[0]
    
    # Graceful degradation uses min=0.0, max=1.0, so projected remains 0.84
    projected = ProjectionEngine().project(candidate, cfg)
    assert projected["confidence"] == 0.84