from __future__ import annotations

import re
from datetime import datetime, timezone


# Only services explicitly required by features.
SERVICE_KEYWORDS = {
    "Auth Module": ["auth", "authentication", "login", "sign-in", "password"],
    "Tenant Service": ["multi-tenant", "tenant"],
    "Event Service": ["event", "schedule"],
    "Registration Service": ["registration", "sign-up", "onboarding"],
    "Payment Service": ["payment", "billing", "checkout", "pricing"],
    "Reporting Service": ["report", "reporting", "dashboard", "insight", "metrics"],
}


MODEL_MAP = {
    "Auth Module": ["User"],
    "Tenant Service": ["Tenant"],
    "Event Service": ["Event"],
    "Registration Service": ["Registration"],
    "Payment Service": ["Payment"],
    "Reporting Service": [],
}


ROUTE_MAP = {
    "Auth Module": ["/auth/login", "/auth/logout", "/auth/refresh"],
    "Tenant Service": ["/tenants", "/tenants/{id}"],
    "Registration Service": ["/registrations", "/registrations/{id}"],
    "Event Service": ["/events", "/events/{id}"],
    "Payment Service": ["/payments", "/payments/{id}"],
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_daily_scale(metrics: list[str]) -> int | None:
    max_val = None
    for metric in metrics:
        text = _normalize(metric)
        if "per day" in text or "daily" in text:
            match = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
            if match:
                value = int(match.group(1).replace(",", ""))
                max_val = value if max_val is None else max(max_val, value)
    return max_val


def _classify_services(features: list[str]) -> list[str]:
    services: set[str] = set()
    for feat in features:
        text = _normalize(feat)
        for service, keywords in SERVICE_KEYWORDS.items():
            if any(k in text for k in keywords):
                services.add(service)
    return sorted(services)


def _detect_multi_tenant(prd: dict) -> bool:
    features = prd.get("desired_features") or []
    arch = prd.get("architecture_decisions") or {}
    text = " ".join(features + [str(v) for v in arch.values() if v])
    return "multi-tenant" in _normalize(text) or "tenant" in _normalize(text)


def _init_model_registry(services: list[str]) -> dict[str, set[str]]:
    registry: dict[str, set[str]] = {}
    for service in services:
        for model in MODEL_MAP.get(service, []):
            registry.setdefault(model, set())
    return registry


def _apply_multi_tenant_fields(models: dict[str, set[str]], multi_tenant: bool) -> None:
    if not multi_tenant:
        return
    for name, fields in models.items():
        if name in {"Event", "User"}:
            fields.add("tenant_id")


def _derive_routes(services: list[str]) -> list[str]:
    routes: list[str] = []
    for service in services:
        if service == "Reporting Service":
            continue
        routes.extend(ROUTE_MAP.get(service, []))
    return sorted(set(routes))


def _infra_layout(tech_stack: dict) -> list[str]:
    frontend = tech_stack.get("frontend_choice", "Not specified")
    backend = tech_stack.get("backend_choice", "Not specified")
    database = tech_stack.get("database_choice", "Not specified")
    infra = tech_stack.get("infra_region") or tech_stack.get("infra_provider") or "Not specified"
    deployment = tech_stack.get("deployment_strategy", "Not specified")
    return [
        f"Frontend: {frontend}",
        f"Backend: {backend}",
        f"Database: {database}",
        f"Infrastructure: {infra}",
        f"Deployment: {deployment}",
    ]


def _security_section(non_functional: dict) -> list[str]:
    security = non_functional.get("security_requirements", "Not specified")
    return [f"Security requirements: {security}"]


def generate_system_design(prd: dict) -> str:
    features = prd.get("desired_features") or []
    arch = prd.get("architecture_decisions") or {}
    services = _classify_services(features)
    multi_tenant = _detect_multi_tenant(prd)
    has_auth = bool(arch.get("authentication_strategy"))
    if has_auth and "Auth Module" not in services:
        services.append("Auth Module")
    if multi_tenant and "Tenant Service" not in services:
        services.append("Tenant Service")
    services = sorted(set(services))

    normalized_features = _normalize(" ".join(features))
    auth_text = _normalize(str(arch.get("authentication_strategy") or ""))

    models_registry = _init_model_registry(services)
    if "pricing tier" in normalized_features or "tiered pricing" in normalized_features:
        models_registry.setdefault("TicketTier", set()).add("event_id")
    if "Registration Service" in services:
        reg_fields = models_registry.setdefault("Registration", set())
        reg_fields.add("qr_code")
        reg_fields.add("ticket_tier_id")
        reg_fields.add("attendee_email")
        reg_fields.add("payment_status")
        reg_fields.add("created_at")
    if has_auth and "credential" in auth_text:
        models_registry.setdefault("UserCredential", set())

    _apply_multi_tenant_fields(models_registry, multi_tenant)
    if "Registration" in models_registry:
        models_registry["Registration"].add("event_id")
    if "Payment" in models_registry:
        models_registry["Payment"].add("registration_id")

    models = []
    for name in sorted(models_registry.keys()):
        fields = sorted(models_registry[name])
        if fields:
            models.append(f"{name} ({', '.join(fields)})")
        else:
            models.append(name)
    routes = _derive_routes(services)
    if "Reporting Service" in services:
        routes.extend(_reporting_routes(features))
        routes = sorted(set(routes))
    scale = _extract_daily_scale(prd.get("success_metrics") or [])

    infra = _infra_layout(prd.get("tech_stack") or {})
    security = _security_section(prd.get("non_functional") or {})

    date_str = _utcnow().strftime("%B %d, %Y")
    lines = [
        "# System Design",
        f"{prd.get('project_name', 'Unnamed Project')} — Version 1.0",
        f"Date: {date_str}",
        "",
        "## 1. Service Decomposition",
        "\n".join([f"- {s}" for s in services]) or "- None",
        "",
        "## 2. Data Models",
        "\n".join([f"- {m}" for m in models]) or "- None",
        "",
        "## 3. API Routes",
        "\n".join([f"- {r}" for r in routes]) or "- None",
        "",
        "## 4. Infrastructure Layout",
        "\n".join([f"- {i}" for i in infra]),
        "",
        "## 5. Security Design",
        "\n".join([f"- {s}" for s in security]),
    ]

    if scale and scale > 5000:
        lines.extend(
            [
                "",
                "## 6. Horizontal Scaling",
                "- Stateless API layer",
                "- Load balancing with autoscaling",
                "- Database read replicas if required",
            ]
        )

    lines.append("\n---END OF SYSTEM DESIGN---")
    return "\n".join(lines)
def _reporting_routes(features: list[str]) -> list[str]:
    text = _normalize(" ".join(features))
    finance_keywords = ["finance", "expense", "spend", "budget", "category"]
    event_keywords = ["event", "registration", "ticket", "attendance", "attendee"]
    routes: list[str] = []
    if any(k in text for k in finance_keywords):
        routes.extend(["/reports/by-category", "/reports/by-date"])
    if any(k in text for k in event_keywords):
        routes.extend(["/reports/revenue", "/reports/attendance", "/reports/registrations"])
    return sorted(set(routes))
