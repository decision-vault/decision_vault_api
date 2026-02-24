from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import settings
from app.services.llm_usage_service import log_llm_usage
from app.services.token_limiter import TokenLimiter

SYSTEM_RULES = (
    "You are generating CONTENT ONLY for a predefined SDD schema. "
    "Do not output markdown headers, section numbers, code fences, or wrapper prose. "
    "Output valid JSON only. "
    "Only use provided information. Do not invent statistics, integrations, or technologies not present in input. "
    "If data is missing for any field, output exactly: 'Insufficient information provided.'"
)

STYLE_RULES = (
    "Write implementation-ready, detailed technical content. "
    "Narrative fields should be 2-4 paragraphs. "
    "List fields should include 5-10 concrete bullets when input supports it."
)

MAX_INPUT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 1200


class Stage1Output(BaseModel):
    executive_summary: str
    purpose: str
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    related_documents: list[str] = Field(default_factory=list)
    architecture_overview: str
    architecture_principles: list[str] = Field(default_factory=list)
    deployment_overview: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        return _normalize_stage_lists(
            value,
            {"scope_in", "scope_out", "related_documents", "architecture_principles", "deployment_overview"},
        )


class Stage2Output(BaseModel):
    data_model_overview: str
    schema_tenant: str
    schema_user: str
    schema_project: str
    schema_decision: str
    data_access_patterns: list[str] = Field(default_factory=list)
    api_design_overview: str
    api_endpoints: list[str] = Field(default_factory=list)
    middleware_pipeline: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        normalized = _normalize_stage_lists(
            value,
            {"data_access_patterns", "api_endpoints", "middleware_pipeline"},
        )
        if isinstance(normalized, dict):
            for field in ("data_model_overview", "api_design_overview"):
                raw = normalized.get(field)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    continue
                if isinstance(raw, dict):
                    description = str(raw.get("description") or "").strip()
                    bullets = []
                    for key in ("details", "items", "points", "bullets", "notes"):
                        val = raw.get(key)
                        if isinstance(val, list):
                            bullets.extend([str(v).strip() for v in val if str(v).strip()])
                    if description and bullets:
                        normalized[field] = f"{description}\n" + "\n".join([f"- {b}" for b in bullets])
                    elif description:
                        normalized[field] = description
                    elif bullets:
                        normalized[field] = "\n".join([f"- {b}" for b in bullets])
                    else:
                        normalized[field] = "\n".join([f"{k}: {v}" for k, v in raw.items()])
                    continue
                if isinstance(raw, list):
                    normalized[field] = "\n".join([str(item).strip() for item in raw if str(item).strip()])
                    continue
                normalized[field] = str(raw)

            for field in ("schema_tenant", "schema_user", "schema_project", "schema_decision"):
                raw = normalized.get(field)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    continue
                if isinstance(raw, dict):
                    normalized[field] = "\n".join([f"{k}: {v}" for k, v in raw.items()])
                    continue
                if isinstance(raw, list):
                    normalized[field] = "\n".join([str(item).strip() for item in raw if str(item).strip()])
                    continue
                normalized[field] = str(raw)
        return normalized


class Stage3Output(BaseModel):
    slack_oauth_flow: list[str] = Field(default_factory=list)
    slack_event_flow: list[str] = Field(default_factory=list)
    slack_permissions: list[str] = Field(default_factory=list)
    slack_privacy: list[str] = Field(default_factory=list)
    security_auth: list[str] = Field(default_factory=list)
    security_rbac: list[str] = Field(default_factory=list)
    security_data: list[str] = Field(default_factory=list)
    security_audit: list[str] = Field(default_factory=list)
    performance_targets: list[str] = Field(default_factory=list)
    scaling_plan: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        return _normalize_stage_lists(
            value,
            {
                "slack_oauth_flow",
                "slack_event_flow",
                "slack_permissions",
                "slack_privacy",
                "security_auth",
                "security_rbac",
                "security_data",
                "security_audit",
                "performance_targets",
                "scaling_plan",
            },
        )


class Stage4Output(BaseModel):
    monitoring_logging: list[str] = Field(default_factory=list)
    alerting_metrics: list[str] = Field(default_factory=list)
    health_checks: list[str] = Field(default_factory=list)
    cicd_pipeline: list[str] = Field(default_factory=list)
    migration_strategy: list[str] = Field(default_factory=list)
    dr_plan: list[str] = Field(default_factory=list)
    testing_strategy: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        return _normalize_stage_lists(
            value,
            {
                "monitoring_logging",
                "alerting_metrics",
                "health_checks",
                "cicd_pipeline",
                "migration_strategy",
                "dr_plan",
                "testing_strategy",
                "glossary",
                "references",
            },
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _provider_config() -> tuple[str, str, str | None, str]:
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "lmstudio":
        return (
            settings.lmstudio_model or settings.llm_model,
            settings.llm_api_key or "lm-studio",
            settings.lmstudio_base_url,
            "lmstudio",
        )
    if provider == "huggingface":
        return (
            settings.hf_openai_model or settings.llm_model,
            settings.hf_api_token,
            settings.hf_router_base_url,
            "huggingface",
        )
    return (settings.llm_model, settings.llm_api_key, settings.llm_base_url, "default")


def _normalize_openai_base_url(base_url: str | None, provider: str) -> str | None:
    if not base_url:
        return base_url
    normalized = base_url.rstrip("/")
    if provider == "lmstudio":
        if normalized.endswith("/api/v1"):
            normalized = normalized[: -len("/api/v1")] + "/v1"
        elif not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
    return normalized


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _repair_json(raw: str) -> str:
    text = raw.strip()
    if text.count("{") > text.count("}"):
        text = text + ("}" * (text.count("{") - text.count("}")))
    return text


def _split_to_list(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.lower() == "insufficient information provided.":
        return []
    parts = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        if "," in line and len(line) > 40:
            chunks = [c.strip() for c in line.split(",") if c.strip()]
            parts.extend(chunks)
        else:
            parts.append(line)
    return parts


def _normalize_stage_lists(value: Any, list_fields: set[str]) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field in list_fields:
        raw = normalized.get(field)
        if raw is None:
            continue
        if isinstance(raw, list):
            normalized[field] = [str(v).strip() for v in raw if str(v).strip()]
            continue
        if isinstance(raw, dict):
            normalized[field] = [f"{k}: {v}" for k, v in raw.items()]
            continue
        if isinstance(raw, str):
            normalized[field] = _split_to_list(raw)
            continue
        normalized[field] = [str(raw).strip()]
    return normalized


def _parse_json(raw: str) -> dict[str, Any]:
    candidate = _repair_json(_extract_json_candidate(raw))
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = ast.literal_eval(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object")
    return parsed


def _safe_bullets(items: list[str] | None) -> str:
    if not items:
        return "- Insufficient information provided."
    cleaned = [str(i).strip() for i in items if str(i).strip()]
    if not cleaned:
        return "- Insufficient information provided."
    return "\n".join([f"- {i}" for i in cleaned])


def _sql_block(content: str) -> str:
    return f"```sql\n{content.strip()}\n```"


def _bounded_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False)
    if _estimate_tokens(raw) <= MAX_INPUT_TOKENS:
        return payload

    compressed: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            compressed[key] = TokenLimiter.compress_text(value, max_tokens=200)
        elif isinstance(value, list):
            compressed[key] = [TokenLimiter.compress_text(str(v), max_tokens=50) for v in value[:30]]
        elif isinstance(value, dict):
            compressed[key] = {k: TokenLimiter.compress_text(str(v), max_tokens=50) for k, v in value.items()}
        else:
            compressed[key] = value

    if _estimate_tokens(json.dumps(compressed, ensure_ascii=False)) > MAX_INPUT_TOKENS:
        raise ValueError("SDD input exceeds token budget")
    return compressed


async def _invoke_llm(prompt: str, output_tokens: int) -> tuple[str, int, str]:
    model_name, api_key, base_url, provider = _provider_config()
    if not api_key:
        raise ValueError("LLM API key not configured")
    normalized_base_url = _normalize_openai_base_url(base_url, provider)
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        top_p=0.8,
        max_tokens=output_tokens,
        api_key=api_key,
        base_url=normalized_base_url,
    )
    msg = await llm.ainvoke(prompt)
    text = (getattr(msg, "content", "") or "").strip()
    meta = getattr(msg, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    tokens_used = int(usage.get("total_tokens") or 0) or _estimate_tokens(prompt + "\n" + text)
    return text, max(tokens_used, 1), model_name


async def _run_stage(
    *,
    stage_name: str,
    schema: type[BaseModel],
    instruction: str,
    payload: dict[str, Any],
    tenant_id: str,
    progress_cb=None,
) -> BaseModel:
    started_at = _utcnow()
    if progress_cb:
        await progress_cb({"stage": stage_name, "status": "running", "started_at": started_at.isoformat()})
    bounded = _bounded_payload(payload)
    input_json = json.dumps(bounded, ensure_ascii=False, indent=2)
    input_tokens = _estimate_tokens(input_json)

    prompt = (
        f"System: {SYSTEM_RULES}\n"
        f"Style: {STYLE_RULES}\n"
        f"Stage: {stage_name}\n"
        f"Task: {instruction}\n"
        f"Allowed keys: {', '.join(schema.model_fields.keys())}\n"
        "Output JSON only.\n\n"
        f"Input:\n{input_json}"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        raw_text, total_tokens, model_name = await _invoke_llm(prompt, MAX_OUTPUT_TOKENS)
        try:
            parsed = _parse_json(raw_text)
            result = schema.model_validate(parsed)
            output_tokens = max(1, total_tokens - input_tokens)
            await log_llm_usage(
                feature=f"sdd_multistep:{stage_name}",
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tenant_id=tenant_id,
                stage_name=stage_name,
                retry_count=attempt,
            )
            if progress_cb:
                completed_at = _utcnow()
                duration = max(0.0, (completed_at - started_at).total_seconds())
                await progress_cb(
                    {
                        "stage": stage_name,
                        "status": "completed",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "retry_count": attempt,
                        "duration_seconds": round(duration, 3),
                        "completed_at": completed_at.isoformat(),
                    }
                )
            return result
        except (ValidationError, ValueError, SyntaxError) as exc:
            last_error = exc
            continue
    if progress_cb:
        await progress_cb({"stage": stage_name, "status": "failed", "error": str(last_error)})
    raise ValueError(f"{stage_name} failed: {last_error}")


def _render_sdd(structured: dict, s1: Stage1Output, s2: Stage2Output, s3: Stage3Output, s4: Stage4Output) -> str:
    project_name = structured.get("project_name") or "DecisionVault"
    date_str = _utcnow().strftime("%B %d, %Y")

    return f"""# System Design Document (SDD)
## {project_name} — Stage 1: Decision History Core

**Version:** 1.0  
**Date:** {date_str}  
**Status:** Design Specification  
**Owner:** Engineering Team  
**Contributors:** Architecture, Backend, Frontend, DevOps

---

## Executive Summary
{s1.executive_summary}

---

## Document Purpose and Scope
### Purpose
{s1.purpose}

### Scope
In Scope:
{_safe_bullets(s1.scope_in)}

Out of Scope:
{_safe_bullets(s1.scope_out)}

### Related Documents
{_safe_bullets(s1.related_documents)}

---

## System Architecture Overview
### High-Level Architecture
{s1.architecture_overview}

### Architecture Principles
{_safe_bullets(s1.architecture_principles)}

### Deployment Architecture
{_safe_bullets(s1.deployment_overview)}

---

## Data Architecture
### Data Model Overview
{s2.data_model_overview}

### Core Database Schema
#### Tenant (Organization)
{_sql_block(s2.schema_tenant)}

#### User
{_sql_block(s2.schema_user)}

#### Project
{_sql_block(s2.schema_project)}

#### Decision (Core Entity)
{_sql_block(s2.schema_decision)}

### Data Access Patterns and Optimization
{_safe_bullets(s2.data_access_patterns)}

---

## Component Architecture
### API Layer Design
{s2.api_design_overview}

### API Endpoint Specifications
{_safe_bullets(s2.api_endpoints)}

### Middleware Pipeline
{_safe_bullets(s2.middleware_pipeline)}

---

## Slack Integration Architecture
### OAuth Flow
{_safe_bullets(s3.slack_oauth_flow)}

### Event Subscription Flow
{_safe_bullets(s3.slack_event_flow)}

### Slack Bot Permissions Required
{_safe_bullets(s3.slack_permissions)}

### Data Privacy
{_safe_bullets(s3.slack_privacy)}

---

## Security Architecture
### Authentication System
{_safe_bullets(s3.security_auth)}

### Authorization and RBAC
{_safe_bullets(s3.security_rbac)}

### Data Security
{_safe_bullets(s3.security_data)}

### Audit Logging
{_safe_bullets(s3.security_audit)}

---

## Performance and Scalability
### Performance Requirements
{_safe_bullets(s3.performance_targets)}

### Scalability Design
{_safe_bullets(s3.scaling_plan)}

---

## Monitoring and Observability
### Logging Strategy
{_safe_bullets(s4.monitoring_logging)}

### Metrics and Alerting
{_safe_bullets(s4.alerting_metrics)}

### Health Checks and Circuit Breakers
{_safe_bullets(s4.health_checks)}

---

## Deployment and Operations
### CI/CD Pipeline
{_safe_bullets(s4.cicd_pipeline)}

### Database Migration Strategy
{_safe_bullets(s4.migration_strategy)}

### Disaster Recovery
{_safe_bullets(s4.dr_plan)}

---

## Testing Strategy
{_safe_bullets(s4.testing_strategy)}

---

## Appendix
### Glossary
{_safe_bullets(s4.glossary)}

### References
{_safe_bullets(s4.references)}

---

## Document Change Log
| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | {date_str} | Engineering Team | Initial system design document for Stage 1 |

---

**End of Document**
"""


async def generate_system_design(prd: dict, tenant_id: str, progress_cb=None) -> str:
    stage1 = await _run_stage(
        stage_name="sdd_stage_1_context",
        schema=Stage1Output,
        instruction=(
            "Fill executive summary, purpose/scope, architecture overview, principles, and deployment overview. "
            "Keep it implementation-oriented and detailed."
        ),
        payload={
            "project_name": prd.get("project_name"),
            "problem_statement": prd.get("problem_statement"),
            "target_users": prd.get("target_users"),
            "desired_features": prd.get("desired_features"),
            "success_metrics": prd.get("success_metrics"),
            "constraints": (prd.get("constraints") or {}).get("hard_constraints"),
            "out_of_scope": prd.get("out_of_scope"),
            "tech_stack": prd.get("tech_stack"),
        },
        tenant_id=tenant_id,
        progress_cb=progress_cb,
    )

    stage2 = await _run_stage(
        stage_name="sdd_stage_2_data_api",
        schema=Stage2Output,
        instruction=(
            "Fill data model and API design. SQL fields must be plain SQL text only (no markdown fences). "
            "Generate endpoint and middleware details aligned to the given features."
        ),
        payload={
            "project_name": prd.get("project_name"),
            "desired_features": prd.get("desired_features"),
            "architecture_decisions": prd.get("architecture_decisions"),
            "tech_stack": prd.get("tech_stack"),
            "constraints": (prd.get("constraints") or {}).get("hard_constraints"),
        },
        tenant_id=tenant_id,
        progress_cb=progress_cb,
    )

    stage3 = await _run_stage(
        stage_name="sdd_stage_3_security_performance",
        schema=Stage3Output,
        instruction=(
            "Fill Slack integration, security architecture, RBAC, audit logging, performance targets, and scaling plan."
        ),
        payload={
            "project_name": prd.get("project_name"),
            "desired_features": prd.get("desired_features"),
            "architecture_decisions": prd.get("architecture_decisions"),
            "non_functional": prd.get("non_functional"),
            "success_metrics": prd.get("success_metrics"),
        },
        tenant_id=tenant_id,
        progress_cb=progress_cb,
    )

    stage4 = await _run_stage(
        stage_name="sdd_stage_4_ops_quality",
        schema=Stage4Output,
        instruction=(
            "Fill monitoring/observability, CI/CD, migration, DR plan, testing strategy, glossary, and references."
        ),
        payload={
            "project_name": prd.get("project_name"),
            "desired_features": prd.get("desired_features"),
            "tech_stack": prd.get("tech_stack"),
            "non_functional": prd.get("non_functional"),
            "constraints": (prd.get("constraints") or {}).get("hard_constraints"),
        },
        tenant_id=tenant_id,
        progress_cb=progress_cb,
    )

    return _render_sdd(prd, stage1, stage2, stage3, stage4)
