from app.services.system_design_service import generate_system_design


def test_system_design_includes_services_models_and_scaling():
    prd = {
        "project_name": "EventFlow",
        "desired_features": [
            "Online ticket payments",
            "Multi-tenant organization data",
            "Email reminders",
        ],
        "architecture_decisions": {"authorization_rbac_model": "RBAC", "authentication_strategy": "Email login"},
        "tech_stack": {
            "frontend_choice": "Next.js",
            "backend_choice": "FastAPI",
            "database_choice": "PostgreSQL",
            "infra_provider": "AWS",
            "deployment_strategy": "Autoscaling",
        },
        "non_functional": {"security_requirements": "TLS + AES-256"},
        "success_metrics": ["Support 10,000 registrations per day"],
    }

    output = generate_system_design(prd)

    assert "Payment Service" in output
    assert "Auth Module" in output
    assert "Tenant" in output
    assert "Payment (tenant_id)" not in output
    assert "Tenant (tenant_id)" not in output
    assert "## 6. Horizontal Scaling" in output
    assert "Infrastructure: AWS" in output


def test_system_design_includes_routes_for_services():
    prd = {
        "project_name": "EventFlow",
        "desired_features": ["Event scheduling"],
        "architecture_decisions": {"authentication_strategy": "Email login"},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "/auth/login" in output
    assert "/events" in output


def test_system_design_ticket_tiers_and_qr_code_fields():
    prd = {
        "project_name": "EventFlow",
        "desired_features": ["Tiered pricing", "Online registration"],
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "TicketTier (event_id)" in output
    assert "Registration (attendee_email, created_at, payment_status, qr_code, ticket_tier_id)" in output


def test_system_design_report_model_only_when_explicit():
    prd = {
        "project_name": "EventFlow",
        "desired_features": ["Dashboard insights"],
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "Reporting Service" in output
    assert "Report" not in output
    assert "/reports/by-date" not in output
    assert "/reports/by-category" not in output


def test_system_design_usercredential_only_when_explicit():
    prd = {
        "project_name": "EventFlow",
        "desired_features": ["Event scheduling"],
        "architecture_decisions": {"authentication_strategy": "Email login"},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "UserCredential" not in output


def test_system_design_usercredential_when_password_auth():
    prd = {
        "project_name": "EventFlow",
        "desired_features": ["Event scheduling"],
        "architecture_decisions": {"authentication_strategy": "Email and password"},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "UserCredential" not in output


def test_system_design_reporting_routes_by_domain():
    prd = {
        "project_name": "SpendWise",
        "desired_features": ["Expense tracking", "Monthly reports", "Spending by category"],
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "/reports/by-category" in output
    assert "/reports/by-date" in output
    assert "/reports/revenue" not in output


def test_system_design_uses_infra_region_or_provider():
    prd = {
        "project_name": "EventFlow",
        "desired_features": [],
        "tech_stack": {"infra_region": "us-east-1"},
        "non_functional": {},
        "success_metrics": [],
    }

    output = generate_system_design(prd)

    assert "Infrastructure: us-east-1" in output
