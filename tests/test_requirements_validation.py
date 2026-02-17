from app.services.requirements_service import (
    deep_merge_structured,
    is_low_quality,
    sanitize_field,
    validate_structured,
)


def test_dirty_merge_prevention():
    existing = {"tech_stack": {"database_choice": "MongoDB"}}
    answers = {"tech_stack.database_choice": "PostgreSQL"}
    merged = deep_merge_structured(existing, answers)
    assert merged["tech_stack"]["database_choice"] == "PostgreSQL"


def test_multi_sentence_rejection():
    value = "First sentence. Second sentence."
    assert is_low_quality(value) is True


def test_field_sanitization():
    value = "will be built using React Native. Backend should be FastAPI."
    cleaned = sanitize_field(value)
    assert cleaned == "React Native"


def test_deduplication_logic():
    structured = {
        "project_name": None,
        "problem_statement": None,
        "target_users": None,
        "desired_features": [],
        "architecture_decisions": {},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": None,
        "constraints": {},
        "out_of_scope": None,
    }
    missing, low_quality = validate_structured(structured)
    assert len(missing) == len(set(missing))
    assert len(low_quality) == len(set(low_quality))


def test_ready_for_prd_gating():
    structured = {
        "project_name": "Demo",
        "problem_statement": "Problem",
        "target_users": ["Admins"],
        "desired_features": ["Feature A"],
        "architecture_decisions": {
            "authentication_strategy": "OAuth",
            "authorization_rbac_model": "RBAC",
            "data_sync_strategy": "Sync",
            "offline_support": "No",
            "currency_handling": "USD",
            "multi_platform_support": "iOS/Android",
            "monitoring_and_logging": "Sentry",
        },
        "tech_stack": {
            "frontend_choice": "React",
            "backend_choice": "FastAPI",
            "database_choice": "PostgreSQL",
            "infra_region": "us-east-1",
            "deployment_strategy": "Docker",
        },
        "non_functional": {
            "security_requirements": "AES-256",
            "performance_goals": "p95 < 200ms",
            "compliance_requirements": "SOC2",
        },
        "success_metrics": ["Retention > 30%"],
        "constraints": {"hard_constraints": ["Budget"]},
        "out_of_scope": ["Web app"],
    }
    missing, low_quality = validate_structured(structured)
    assert missing == []
    assert low_quality == []
