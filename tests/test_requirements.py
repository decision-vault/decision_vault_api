from app.schemas.requirements import PRDSchema
from app.services.requirements_service import _missing_fields


def test_missing_fields_detection():
    structured = {
        "project_name": "Demo",
        "problem_statement": "",
        "target_users": [],
        "desired_features": ["A"],
        "architecture_decisions": {"authentication_strategy": None},
        "tech_stack": {"frontend_choice": "React"},
        "non_functional": {"security_requirements": None},
        "success_metrics": [],
        "constraints": {"hard_constraints": []},
        "out_of_scope": [],
    }
    missing = _missing_fields(structured)
    assert "problem_statement" in missing
    assert "target_users" in missing
    assert "architecture_decisions.authentication_strategy" in missing
    assert "non_functional.security_requirements" in missing
    assert "success_metrics" in missing
    assert "constraints.hard_constraints" in missing
    assert "out_of_scope" in missing


def test_prd_schema_validation():
    prd = PRDSchema(
        project_name="Demo",
        problem_statement="Problem",
        target_users=["Admins"],
        desired_features=["Feature A"],
        architecture_decisions={
            "authentication_strategy": "OAuth",
            "authorization_rbac_model": "RBAC",
            "data_sync_strategy": "Sync",
            "offline_support": "No",
            "currency_handling": "USD",
            "multi_platform_support": "iOS/Android",
            "monitoring_and_logging": "Sentry",
        },
        tech_stack={
            "frontend_choice": "React Native",
            "backend_choice": "FastAPI",
            "database_choice": "MongoDB",
            "infra_region": "us-east-1",
            "deployment_strategy": "Docker",
        },
        non_functional={
            "security_requirements": "AES-256",
            "performance_goals": "p95 < 200ms",
            "compliance_requirements": "SOC2",
        },
        success_metrics=["Retention > 30%"],
        constraints={"hard_constraints": ["Budget"]},
        out_of_scope=["Web app"],
    )
    assert prd.project_name == "Demo"
