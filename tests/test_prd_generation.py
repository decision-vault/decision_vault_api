from app.services.prd_service import generate_prd, normalize_structured_for_prd


def test_prd_contains_headers():
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
    text = generate_prd(structured)
    assert "# Product Requirements Document (PRD)" in text
    assert "## Executive Summary" in text
    assert "---END OF PRD---" in text


def test_out_of_scope_rendered():
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
    text = generate_prd(structured)
    assert "## Out of Scope" in text
    assert "- Web app" in text


def test_feature_fragmentation_repair():
    structured = {
        "project_name": "Demo",
        "problem_statement": "Problem",
        "target_users": ["Admins"],
        "desired_features": ["Add", "edit", "and delete expenses", "title", "amount", "category"],
        "architecture_decisions": {
            "authentication_strategy": "OAuth",
            "authorization_rbac_model": "RBAC",
            "data_sync_strategy": "Sync",
            "offline_support": "No",
            "currency_handling": "USD",
            "multi_platform_support": "mobile application",
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
        "success_metrics": ["90% task success", "5", "000 daily active users"],
        "constraints": {"hard_constraints": ["Budget"]},
        "out_of_scope": ["Web app"],
    }
    normalized = normalize_structured_for_prd(structured)
    assert any("Add, edit" in f for f in normalized["desired_features"])
    assert any("5,000 daily active users" in m for m in normalized["success_metrics"])
    assert normalized["architecture_decisions"]["multi_platform_support"] == "Mobile (iOS and Android)"
    # KPI contamination is moved out of data_sync_strategy
    structured["architecture_decisions"]["data_sync_strategy"] = "Maintain 99% sync within three months"
    normalized = normalize_structured_for_prd(structured)
    assert normalized["architecture_decisions"]["data_sync_strategy"] is None
