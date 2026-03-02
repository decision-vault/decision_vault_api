from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.db.mongo import get_db
import re

from app.services.requirements_service import validate_structured


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bullets(items: list[str] | None) -> str:
    if not items:
        return "- None specified"
    return "\n".join([f"- {item}" for item in items])

def _group_features(features: list[str]) -> dict:
    groups = {
        "Core Features": [],
        "User & Access": [],
        "Reporting & Analytics": [],
        "Integrations": [],
        "Other": [],
    }
    for feat in features:
        lower = feat.lower()
        if any(k in lower for k in ["user", "role", "auth", "login", "permission"]):
            groups["User & Access"].append(feat)
        elif any(k in lower for k in ["report", "analytics", "dashboard", "insight", "metrics"]):
            groups["Reporting & Analytics"].append(feat)
        elif any(k in lower for k in ["integration", "webhook", "slack", "api", "export"]):
            groups["Integrations"].append(feat)
        elif any(k in lower for k in ["sync", "offline", "capture", "decision", "create", "edit"]):
            groups["Core Features"].append(feat)
        else:
            groups["Other"].append(feat)
    return {k: v for k, v in groups.items() if v}


def _risk_table(risks: list[str] | None) -> str:
    if not risks:
        risks = ["Risks will be identified during technical design phase."]
    rows = "\n".join([f"| {r} | TBD |" for r in risks])
    return "| Risk | Mitigation |\n|------|------------|\n" + rows
def normalize_structured_for_prd(structured: dict) -> dict:
    cleaned = {**structured}

    # A) Merge fragmented feature tokens
    features = list(cleaned.get("desired_features") or [])
    merged_features: list[str] = []
    i = 0
    descriptors = {"title", "amount", "category", "date", "note", "optional note"}
    while i < len(features):
        cur = features[i].strip()
        if i + 1 < len(features):
            nxt = features[i + 1].strip()
            if len(cur) < 15 and len(nxt) < 15:
                cur = f"{cur}, {nxt}"
                i += 1
            if nxt.lower().startswith("and "):
                cur = f"{cur} {nxt}"
                i += 1
            if nxt.lower() in descriptors:
                cur = f"{cur} including {nxt}"
                i += 1
        # Merge short fragments into previous feature
        if len(cur) < 10 and merged_features:
            merged_features[-1] = f"{merged_features[-1]} {cur}".strip()
        else:
            merged_features.append(cur)
        i += 1

    # D) Ensure meaningful feature length
    cleaned["desired_features"] = [f for f in merged_features if len(f) >= 10]

    # B) Fix split numeric values in success_metrics
    metrics = list(cleaned.get("success_metrics") or [])
    fixed_metrics: list[str] = []
    i = 0
    while i < len(metrics):
        cur = metrics[i].strip()
        if i + 1 < len(metrics):
            nxt = metrics[i + 1].strip()
            if re.match(r"^\d+$", cur) and re.match(r"^\d{3,}", nxt):
                combined = f"{cur},{nxt}"
                if i + 2 < len(metrics):
                    combined = f"{combined} {metrics[i + 2].strip()}"
                    i += 1
                cur = combined
                i += 1
            elif re.match(r"^\d+$", cur) and nxt and re.match(r"^\d", nxt):
                cur = f"{cur}{nxt}"
                i += 1
        fixed_metrics.append(cur)
        i += 1
    cleaned["success_metrics"] = fixed_metrics

    # C) KPI contamination detection
    arch = cleaned.get("architecture_decisions", {})
    sync = arch.get("data_sync_strategy")
    if isinstance(sync, str):
        contamination = any(token in sync for token in ["%", "daily active users", "within three months", "DAU"])
        if contamination:
            metrics = list(cleaned.get("success_metrics") or [])
            metrics.append(sync)
            cleaned["success_metrics"] = metrics
            arch["data_sync_strategy"] = None
    cleaned["architecture_decisions"] = arch

    # E) Normalize multi-platform support
    if arch.get("multi_platform_support") == "mobile application":
        arch["multi_platform_support"] = "Mobile (iOS and Android)"

    return cleaned

def generate_prd(structured: dict) -> str:
    normalized = normalize_structured_for_prd(structured)
    missing, low_quality = validate_structured(normalized)
    if missing or low_quality:
        raise ValueError("Requirement not complete. Cannot generate PRD.")

    date_str = _utcnow().strftime("%B %d, %Y")
    project_name = normalized.get("project_name", "Unnamed Project")
    problem = normalized.get("problem_statement", "")
    users = normalized.get("target_users", [])
    features = normalized.get("desired_features", [])
    arch = normalized.get("architecture_decisions", {})
    tech = normalized.get("tech_stack", {})
    nonfunc = normalized.get("non_functional", {})
    success = normalized.get("success_metrics", [])
    constraints = normalized.get("constraints", {}).get("hard_constraints", [])
    out_of_scope = normalized.get("out_of_scope", [])
    risks = normalized.get("risks", [])

    feature_groups = _group_features(features)
    feature_sections = []
    for group, items in feature_groups.items():
        feature_sections.append(f"### {group}\n{_bullets(items)}")
    feature_paragraphs = "\n\n".join(feature_sections) if feature_sections else "- None specified"
    metrics_paragraphs = "\n".join([f"- {m}" for m in success]) if success else "- None specified"

    content = f"""# Product Requirements Document (PRD)
{project_name} — Version 1.0
Date: {date_str}
Status: Draft for Development

## 1. Executive Summary
{problem}

## 2. Problem Statement
{problem}

## 3. Target Users
{_bullets(users)}

## 4. Objectives and Success Metrics
{metrics_paragraphs}

## 5. Feature Overview
{feature_paragraphs}

## 6. Architecture Decisions
### Authentication Strategy
{arch.get("authentication_strategy", "Not specified")}

### Authorization / RBAC Model
{arch.get("authorization_rbac_model", "Not specified")}

### Data Sync Strategy
{arch.get("data_sync_strategy", "Not specified")}

### Offline Support
{arch.get("offline_support", "Not specified")}

### Currency Handling
{arch.get("currency_handling", "Not specified")}

### Multi-Platform Support
{arch.get("multi_platform_support", "Not specified")}

### Monitoring and Logging
{arch.get("monitoring_and_logging", "Not specified")}

## 7. Technical Architecture
- Frontend: {tech.get("frontend_choice", "Not specified")}
- Backend: {tech.get("backend_choice", "Not specified")}
- Database: {tech.get("database_choice", "Not specified")}
- Infrastructure: {tech.get("infra_region", "Not specified")}
- Deployment Strategy: {tech.get("deployment_strategy", "Not specified")}

## 8. Non-Functional Requirements
- Security: {nonfunc.get("security_requirements", "Not specified")}
- Performance: {nonfunc.get("performance_goals", "Not specified")}
- Compliance: {nonfunc.get("compliance_requirements", "Not specified")}

## 9. Scalability Considerations
- Expected growth aligned to success metrics: {_bullets(success)}
- Stateless service design where applicable
- Horizontal scaling planned via deployment strategy

## 10. Constraints
{_bullets(constraints)}

## 11. Out of Scope
{_bullets(out_of_scope)}

## 12. Risks and Mitigation
{_risk_table(risks)}

## 13. Definition of Done
- All functional features implemented and verified
- Performance targets met under expected load
- Security requirements satisfied and validated
- Monitoring and logging in place
- Documentation updated for operations and support

---END OF PRD---
"""
    return content


async def save_prd(
    intake_id: str,
    content: str,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    base_doc = {
        "intake_id": intake_oid,
        "generated_at": _utcnow(),
        "content": content,
    }
    if tenant_id:
        base_doc["tenant_id"] = ObjectId(tenant_id)
    if project_id:
        base_doc["project_id"] = ObjectId(project_id)

    # Unique index is (intake_id, version), so version must be allocated globally per intake.
    # Retry on duplicate key to survive concurrent inserts.
    for _ in range(3):
        last = await db.prd_documents.find_one({"intake_id": intake_oid}, sort=[("version", -1)])
        next_version = (last.get("version") if last else 0) + 1
        doc = {**base_doc, "version": next_version}
        try:
            await db.prd_documents.insert_one(doc)
            return doc
        except DuplicateKeyError:
            continue

    raise RuntimeError("Failed to save PRD due to concurrent version conflict")
