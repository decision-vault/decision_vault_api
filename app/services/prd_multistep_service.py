from __future__ import annotations

import ast
import json
import logging
import re
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import Counter
from typing import Any, Awaitable, Callable, get_origin

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import settings
from app.schemas.prd_generation import PRDGenerateRequest, PRDMultiStepResponse
from app.services.llm_usage_service import log_llm_usage
from app.services.project_vector_memory_service import (
    retrieve_project_knowledge_chunks,
    store_project_source_text,
    sync_project_knowledge_chunks,
)
from app.services.token_limiter import TokenBudget, TokenLimiter

logger = logging.getLogger("decisionvault.prd.multistep")

SYSTEM_CONTENT_ONLY = (
    "You are generating CONTENT ONLY for a predefined PRD schema. "
    "Do not output section numbers, markdown headers, JSON wrappers, or code fences. "
    "Do not invent sections, integrations, features, statistics, percentages, currency, or technologies not present in input. "
    "Do not change structure. Fill only requested fields. "
    "If input is insufficient for a field, output exactly: 'Insufficient information provided.'"
)

DOC_STYLE_GUIDE = (
    "Return strict JSON that matches the requested schema exactly. "
    "Each field value must be markdown-ready prose (paragraphs and bullet content), or a list of strings. "
    "Do not include markdown headings because the renderer owns headings and numbering. "
    "Do not include escaped newline literals (\\\\n), key commentary, examples, or additional keys."
)

DETAIL_DEPTH_GUIDE = (
    "Write detailed, concrete content suitable for enterprise PRD review. "
    "For narrative fields, provide 2-4 substantive paragraphs. "
    "For list fields, provide 5-10 specific items when input supports it. "
    "Prefer explicit constraints, measurable outcomes, and implementation-ready wording."
)

MAX_STAGE_INPUT_TOKENS = 1500
MAX_STAGE_OUTPUT_TOKENS = 1200

REQUIRED_SECTION_HEADINGS = [
    "# Product Requirements Document (PRD)",
    "## 1. Document header / metadata",
    "### 1.1 Title and project code",
    "### 1.2 Version, author, reviewers, approvers",
    "### 1.3 Date of creation and last update",
    "### 1.4 Applicable teams (Product, Eng, Design, QA, DevOps, Compliance, etc.)",
    "## 2. Introduction & background",
    "### 2.1 Problem statement / context",
    "### 2.2 Why this project is being built (pain points, business need)",
    "### 2.3 Link to strategy / roadmap / OKRs",
    "## 3. Goals & success metrics",
    "### 3.1 Product goals (what \"good\" looks like)",
    "### 3.2 Key success metrics (KPIs)",
    "### 3.3 Go-live / success criteria (what \"done\" means)",
    "## 4. Target users & personas",
    "### 4.1 Who are the main user types (end users, admins, ops, partners)",
    "### 4.2 User personas with roles, goals, and pain points",
    "### 4.3 Market / customer segment overview (optional)",
    "## 5. Scope and boundaries",
    "### 5.1 In-scope features (what this release will cover)",
    "### 5.2 Out-of-scope items (what will be handled later or in another project)",
    "### 5.3 Assumptions about data, integrations, tech stack, timelines",
    "## 6. Features & functional requirements",
    "## 7. Non-functional requirements",
    "### 7.1 Performance (latency, throughput, SLAs)",
    "### 7.2 Reliability / availability (uptime, retry logic)",
    "### 7.3 Security (auth, RBAC, data encryption, PII handling)",
    "### 7.4 Scalability, load limits, horizontal vs vertical",
    "### 7.5 Compliance (GDPR, HIPAA, SOC2, internal policy)",
    "### 7.6 Observability (logging, monitoring, alerting)",
    "### 7.7 Disaster recovery / backup strategy",
    "## 8. User experience and design",
    "### 8.1 Key user flows / task flows",
    "### 8.2 Design links (Figma, Zeplin, sketches, mockups)",
    "### 8.3 UX guidelines / constraints (brand, accessibility, device support)",
    "### 8.4 Copy / localization notes (if applicable)",
    "## 9. Integrations & dependencies",
    "### 9.1 External systems (3rd-party APIs, partners, legacy systems)",
    "### 9.2 Internal systems (CRM, ERP, auth service, data warehouse, etc.)",
    "### 9.3 Data dependencies (feeds, pipelines, CDC, batch vs real-time)",
    "### 9.4 Legal / contractual dependencies (licenses, SLAs, approvals)",
    "## 10. Data model & business rules",
    "### 10.1 High-level entities / tables / schemas",
    "### 10.2 Core business rules",
    "### 10.3 Data retention / purge policies",
    "### 10.4 Reporting / analytics needs (dashboards, exports)",
    "## 11. Release plan & milestones",
    "### 11.1 Release strategy (phased rollout, feature flags, A/B tests)",
    "### 11.2 Key milestones (design freeze, backend ready, UAT, production deploy)",
    "### 11.3 Target release dates",
    "### 11.4 Rollback plan",
    "## 12. Testing & QA",
    "### 12.1 Test scope (manual / automated)",
    "### 12.2 Key test scenarios linked to features",
    "### 12.3 Non-functional test focus (performance, security, soak, chaos, etc.)",
    "### 12.4 UAT / staging environment requirements",
    "## 13. Operations & DevOps",
    "### 13.1 Deployment strategy (canary, blue-green, CI/CD)",
    "### 13.2 Environment requirements (dev, qa, staging, prod, data volumes)",
    "### 13.3 Monitoring / alerting thresholds",
    "### 13.4 Backup / restore / DR process",
    "### 13.5 Required infrastructure (cloud region, nodes, DB size, etc.)",
    "## 14. Risks, assumptions & constraints",
    "### 14.1 Technical risks",
    "### 14.2 Business risks",
    "### 14.3 Schedule / resource constraints",
    "### 14.4 Known open questions / decisions pending",
    "## 15. References & appendices",
    "### 15.1 Related docs",
    "### 15.2 Glossary of terms",
    "### 15.3 Change history / version log",
]

EXPECTED_NUMBER_PATHS = [
    "1.", "1.1", "1.2", "1.3", "1.4",
    "2.", "2.1", "2.2", "2.3",
    "3.", "3.1", "3.2", "3.3",
    "4.", "4.1", "4.2", "4.3",
    "5.", "5.1", "5.2", "5.3",
    "6.", "6.1", "6.1.1", "6.1.2", "6.1.3", "6.1.4", "6.1.5", "6.1.6",
    "6.2", "6.2.1", "6.2.2", "6.2.3", "6.2.4", "6.2.5", "6.2.6",
    "7.", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7",
    "8.", "8.1", "8.2", "8.3", "8.4",
    "9.", "9.1", "9.2", "9.3", "9.4",
    "10.", "10.1", "10.2", "10.3", "10.4",
    "11.", "11.1", "11.2", "11.3", "11.4",
    "12.", "12.1", "12.2", "12.3", "12.4",
    "13.", "13.1", "13.2", "13.3", "13.4", "13.5",
    "14.", "14.1", "14.2", "14.3", "14.4",
    "15.", "15.1", "15.2", "15.3",
]

PRD_STAGE_ORDER = [
    "stage_01_context_snapshot",
    "stage_02_intro_summary",
    "stage_03_doc_metadata",
    "stage_04_intro_problem_context",
    "stage_05_intro_why_build",
    "stage_06_intro_strategy_okrs",
    "stage_07_goals_primary",
    "stage_08_goals_kpis",
    "stage_09_goals_done_criteria",
    "stage_10_users_types",
    "stage_11_users_personas",
    "stage_12_users_market_optional",
    "stage_13_scope_in",
    "stage_14_scope_out",
    "stage_15_scope_assumptions",
    "stage_16_features_user_stories",
    "stage_17_architecture_data_api_integrations",
    "stage_18_ux_design",
    "stage_19_delivery_quality",
    "stage_20_finalize",
]


def _count_heading_lines(md: str, heading: str) -> int:
    pattern = rf"(?m)^{re.escape(heading)}\s*$"
    return len(re.findall(pattern, md))


class SectionContent(BaseModel):
    title: str
    content: str


class Persona(BaseModel):
    name: str
    role: str = "Insufficient information provided."
    description: str
    pain_points: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class UserStory(BaseModel):
    id: str
    description: str = "Insufficient information provided."
    acceptance_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if not normalized.get("description") and normalized.get("title"):
            normalized["description"] = normalized.get("title")
        criteria = normalized.get("acceptance_criteria")
        if isinstance(criteria, str):
            normalized["acceptance_criteria"] = PRDOrchestrator._split_to_list(criteria)
        return normalized


class FeatureDetail(BaseModel):
    feature_name: str = "Insufficient information provided."
    detailed_flows: str = "Insufficient information provided."
    io_validation_rules: str = "Insufficient information provided."
    error_failure_handling: str = "Insufficient information provided."
    priority: str = "Insufficient information provided."

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, value: Any) -> Any:
        # Models sometimes emit these fields as lists of lines; coerce to strings.
        if not isinstance(value, dict):
            return value
        normalized = dict(value)

        def to_text(v: Any) -> str:
            if v is None:
                return "Insufficient information provided."
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                parts = [str(x).strip() for x in v if str(x).strip()]
                if not parts:
                    return "Insufficient information provided."
                if len(parts) == 1:
                    return parts[0]
                return "\n".join([f"- {p}" for p in parts])
            if isinstance(v, dict):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return str(v)

        for k in ["feature_name", "detailed_flows", "io_validation_rules", "error_failure_handling", "priority"]:
            if k in normalized:
                normalized[k] = to_text(normalized.get(k))
        return normalized


class PRDContent(BaseModel):
    executive_summary: str
    core_problem: str
    why_tools_fail: str
    success_meaning: str
    primary_objective: str
    success_metrics: list[str] = Field(default_factory=list)
    leading_indicators: list[str] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    in_scope_features: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)
    feature_details: list[FeatureDetail] = Field(default_factory=list)
    architecture_summary: str
    data_model_summary: str
    api_summary: str
    slack_integration_summary: str
    security_summary: str
    ui_summary: str
    dependencies_summary: str
    non_functional_summary: str
    testing_summary: str
    launch_plan_summary: str
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)

    # Optional fields used by the new outline (kept optional for backward compatibility).
    strategy_okrs: str = "Insufficient information provided."
    user_types: list[str] = Field(default_factory=list)
    market_overview: str = "Insufficient information provided."
    doc_version: str = "1.0"
    doc_author: str = "Product Team"
    doc_reviewers: list[str] = Field(default_factory=lambda: ["Engineering", "Design", "QA"])
    doc_approvers: list[str] = Field(default_factory=lambda: ["Product Leadership"])
    applicable_teams: list[str] = Field(default_factory=lambda: ["Product", "Engineering", "Design", "QA", "DevOps", "Compliance"])
    ops_devops_summary: str = "Insufficient information provided."
    copy_localization_notes: str = "Insufficient information provided."
    target_release_dates: str = "Insufficient information provided."


class Stage01CoreOutput(BaseModel):
    executive_summary: str
    core_problem: str
    why_tools_fail: str
    success_meaning: str


class Stage01ContextOutput(BaseModel):
    context_summary: str


class Stage02IntroSummaryOutput(BaseModel):
    executive_summary: str
    success_meaning: str


class Stage03CoreProblemOutput(BaseModel):
    core_problem: str


class Stage04WhyToolsFailOutput(BaseModel):
    why_tools_fail: str


class Stage05StrategyOkrsOutput(BaseModel):
    strategy_okrs: str


class Stage06PrimaryObjectiveOutput(BaseModel):
    primary_objective: str


class Stage07SuccessMetricsOutput(BaseModel):
    success_metrics: list[str] = Field(default_factory=list)


class Stage08LeadingIndicatorsOutput(BaseModel):
    leading_indicators: list[str] = Field(default_factory=list)


class Stage08KpisOutput(BaseModel):
    success_metrics: list[str] = Field(default_factory=list)
    leading_indicators: list[str] = Field(default_factory=list)


class Stage09DefinitionOfDoneOutput(BaseModel):
    definition_of_done: list[str] = Field(default_factory=list)


class Stage10UserTypesOutput(BaseModel):
    user_types: list[str] = Field(default_factory=list)


class Stage12MarketOverviewOutput(BaseModel):
    market_overview: str


class Stage03DocMetadataOutput(BaseModel):
    doc_version: str = "1.0"
    doc_author: str = "Product Team"
    doc_reviewers: list[str] = Field(default_factory=list)
    doc_approvers: list[str] = Field(default_factory=list)
    applicable_teams: list[str] = Field(default_factory=list)


class Stage10InScopeFeaturesOutput(BaseModel):
    in_scope_features: list[str] = Field(default_factory=list)


class Stage11OutOfScopeOutput(BaseModel):
    out_of_scope: list[str] = Field(default_factory=list)


class Stage12UserStoriesOutput(BaseModel):
    user_stories: list[UserStory] = Field(default_factory=list)
    feature_details: list[FeatureDetail] = Field(default_factory=list)


class Stage15AssumptionsOutput(BaseModel):
    assumptions: list[str] = Field(default_factory=list)


class Stage17ArchitecturePackOutput(BaseModel):
    architecture_summary: str
    data_model_summary: str
    api_summary: str
    slack_integration_summary: str
    security_summary: str
    dependencies_summary: str


class Stage18NonFunctionalOutput(BaseModel):
    non_functional_summary: str


class Stage13ArchitectureSummaryOutput(BaseModel):
    architecture_summary: str


class Stage14DataModelSummaryOutput(BaseModel):
    data_model_summary: str


class Stage15ApiSummaryOutput(BaseModel):
    api_summary: str


class Stage16SlackIntegrationSummaryOutput(BaseModel):
    slack_integration_summary: str


class Stage17SecuritySummaryOutput(BaseModel):
    security_summary: str


class Stage18UiSummaryOutput(BaseModel):
    ui_summary: str
    copy_localization_notes: str = "Insufficient information provided."


class Stage19DeliveryQualityOutput(BaseModel):
    dependencies_summary: str
    non_functional_summary: str
    testing_summary: str
    launch_plan_summary: str
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)

    ops_devops_summary: str = "Insufficient information provided."


class Stage19FinalizeOutput(BaseModel):
    non_functional_summary: str
    testing_summary: str
    launch_plan_summary: str
    ops_devops_summary: str = "Insufficient information provided."
    target_release_dates: str = "Insufficient information provided."
    open_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)


class Stage06ObjectivesOutput(BaseModel):
    primary_objective: str
    success_metrics: list[str] = Field(default_factory=list)
    leading_indicators: list[str] = Field(default_factory=list)


class Stage09PersonasOutput(BaseModel):
    personas: list[Persona] = Field(default_factory=list)


class Stage2Output(BaseModel):
    in_scope_features: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_user_stories(cls, value: Any) -> Any:
        """
        Models sometimes fail to return strict JSON for `user_stories` and instead emit a list of
        lines (markdown-ish). Recover by grouping lines into UserStory objects.
        """
        if not isinstance(value, dict):
            return value
        raw = value.get("user_stories")
        if isinstance(raw, str):
            raw = PRDOrchestrator._split_to_list(raw)
        if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
            normalized = dict(value)
            normalized["user_stories"] = _parse_user_stories_from_lines([str(x) for x in raw])
            return normalized
        return value


def _parse_user_stories_from_lines(lines: list[str]) -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    mode: str = "desc"

    def flush() -> None:
        nonlocal current
        if not current:
            return
        desc = str(current.get("description") or "").strip()
        if not desc:
            current["description"] = "Insufficient information provided."
        criteria = current.get("acceptance_criteria")
        if not isinstance(criteria, list):
            current["acceptance_criteria"] = []
        current["acceptance_criteria"] = [str(x).strip(" -\t") for x in current["acceptance_criteria"] if str(x).strip()]
        stories.append(current)
        current = None

    for raw in lines:
        line = (raw or "").strip()
        if not line or line in {"[", "]"}:
            continue

        m = re.match(r"^\s*(US-\d+\.\d+)\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            flush()
            current = {
                "id": m.group(1).upper(),
                "description": (m.group(2) or "").strip(),
                "acceptance_criteria": [],
            }
            mode = "desc"
            continue

        if not current:
            continue

        if re.search(r"acceptance\s*criteria", line, flags=re.IGNORECASE):
            mode = "ac"
            continue

        bullet = re.sub(r"^\s*[*\-•]\s*", "", line).strip()
        if mode == "ac":
            if bullet:
                current["acceptance_criteria"].append(bullet)
            continue

        # Description: tolerate multi-line narratives and markdown bullet lines.
        fragment = bullet if (line.lstrip().startswith(("*", "-", "•")) and bullet) else line
        if fragment:
            joined = (str(current.get("description") or "") + " " + fragment).strip()
            current["description"] = joined

    flush()
    return stories


class Stage3Output(BaseModel):
    architecture_summary: str
    data_model_summary: str
    api_summary: str
    slack_integration_summary: str
    security_summary: str


class Stage4Output(BaseModel):
    ui_summary: str
    dependencies_summary: str
    non_functional_summary: str
    testing_summary: str
    launch_plan_summary: str
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)


@dataclass
class StageRunResult:
    output: BaseModel
    input_tokens: int
    output_tokens: int
    retry_count: int


class PRDOrchestrator:
    def __init__(
        self,
        tenant_id: str,
        project_id: str | None = None,
        intake_id: str | None = None,
        run_id: str | None = None,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        control_cb: Callable[[], Awaitable[dict[str, bool]] | dict[str, bool] | None] | None = None,
        resume_from_stage: str | None = None,
        stage_cache: dict[str, dict[str, Any]] | None = None,
    ):
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.intake_id = intake_id
        self.run_id = run_id
        self.progress_cb = progress_cb
        self.control_cb = control_cb
        self.total_tokens_used = 0
        self.sections_generated: list[str] = []
        self.retrieved_chunks: list[str] = []
        self.limiter = TokenLimiter(
            TokenBudget(max_input_tokens=MAX_STAGE_INPUT_TOKENS, max_output_tokens=MAX_STAGE_OUTPUT_TOKENS)
        )
        self._pause_emitted = False
        self._resume_from_stage = str(resume_from_stage or "").strip() or None
        self._stage_cache: dict[str, dict[str, Any]] = stage_cache or {}

    def _context_window_tokens(self) -> int:
        _model_name, _api_key, _base_url, provider = self._provider_config()
        if provider == "huggingface":
            return int(getattr(settings, "hf_max_input_tokens", 2048) or 2048)
        return int(getattr(settings, "llm_context_window_tokens", 8192) or 8192)

    def _context_safety_margin(self) -> int:
        return int(getattr(settings, "llm_context_safety_margin_tokens", 256) or 256)

    async def _emit_progress(self, event: dict[str, Any]) -> None:
        if not self.progress_cb:
            return
        maybe = self.progress_cb(event)
        if hasattr(maybe, "__await__"):
            await maybe

    @staticmethod
    def _is_insufficient(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip() == "Insufficient information provided."
        if isinstance(value, list):
            return any(isinstance(item, str) and item.strip() == "Insufficient information provided." for item in value)
        return False

    @classmethod
    def _missing_fields(cls, parsed: BaseModel) -> list[str]:
        missing: list[str] = []
        raw = parsed.model_dump()
        for k, v in raw.items():
            if cls._is_insufficient(v):
                missing.append(str(k))
        return missing

    async def _control_snapshot(self) -> dict[str, Any]:
        if not self.control_cb:
            return {}
        maybe = self.control_cb()
        controls = await maybe if hasattr(maybe, "__await__") else maybe
        return controls or {}

    async def _check_control(self, stage_name: str) -> None:
        if not self.control_cb:
            return
        while True:
            maybe = self.control_cb()
            controls = await maybe if hasattr(maybe, "__await__") else maybe
            controls = controls or {}
            if controls.get("stop"):
                await self._emit_progress({"stage": stage_name, "status": "failed", "error": "Run stopped by user."})
                raise RuntimeError("Run stopped by user.")
            if controls.get("pause"):
                if not self._pause_emitted:
                    self._pause_emitted = True
                    await self._emit_progress({"stage": stage_name, "status": "paused"})
                await asyncio.sleep(1.0)
                continue
            if self._pause_emitted:
                self._pause_emitted = False
                await self._emit_progress({"stage": stage_name, "status": "running"})
            return

    @classmethod
    def _progress_payload_from_model(cls, model: BaseModel) -> dict[str, Any]:
        raw = model.model_dump()

        def shrink(value: Any) -> Any:
            if isinstance(value, str):
                text = value.strip()
                return text if len(text) <= 400 else f"{text[:400]}..."
            if isinstance(value, list):
                return [shrink(v) for v in value[:5]]
            if isinstance(value, dict):
                out: dict[str, Any] = {}
                for k, v in list(value.items())[:8]:
                    out[k] = shrink(v)
                return out
            return value

        return shrink(raw)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return TokenLimiter.estimate_tokens(text)

    @staticmethod
    def _split_to_list(value: str) -> list[str]:
        parts = [p.strip(" -\t") for p in re.split(r"[\n,;]+", value or "")]
        return [p for p in parts if p]

    @staticmethod
    def _sanitize_text(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "Insufficient information provided."
        text = text.replace("\\n", "\n").replace("\\t", " ")
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip().strip('"').strip("'")
        # Remove model-added markdown headers; renderer owns headers/numbering.
        text = "\n".join([line for line in text.splitlines() if not re.match(r"^\s{0,3}#{1,6}\s+", line)])
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text.startswith("{") and text.endswith("}"):
            return "Insufficient information provided."
        return text

    @classmethod
    def _coerce_schema_lists(cls, payload: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
        patched = dict(payload)
        for field_name, field in schema.model_fields.items():
            if field_name not in patched:
                continue
            origin = get_origin(field.annotation)
            if origin is list:
                if isinstance(patched[field_name], str):
                    patched[field_name] = cls._split_to_list(patched[field_name])
                elif isinstance(patched[field_name], dict):
                    patched[field_name] = [
                        f"{str(k).strip()}: {str(v).strip()}"
                        for k, v in patched[field_name].items()
                        if str(k).strip() and str(v).strip()
                    ]
        return patched

    @classmethod
    def _fill_missing_schema_fields(cls, payload: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
        patched = dict(payload)
        for field_name, field in schema.model_fields.items():
            if field_name in patched and patched[field_name] not in (None, ""):
                continue
            origin = get_origin(field.annotation)
            if origin is list:
                patched[field_name] = []
            else:
                patched[field_name] = "Insufficient information provided."
        return patched

    @classmethod
    def _sanitize_obj(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._sanitize_text(value)
        if isinstance(value, list):
            out = [cls._sanitize_obj(v) for v in value]
            return [v for v in out if v not in (None, "", [], {})]
        if isinstance(value, dict):
            return {k: cls._sanitize_obj(v) for k, v in value.items()}
        return value

    @staticmethod
    def _normalize_json_candidate(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _balance_json_like(text: str) -> str:
        stack: list[str] = []
        out: list[str] = []
        in_string = False
        escape = False
        pairs = {"{": "}", "[": "]"}
        for ch in text:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                continue
            if ch in "{[":
                stack.append(pairs[ch])
                out.append(ch)
                continue
            if ch in "}]":
                if stack and ch == stack[-1]:
                    stack.pop()
                    out.append(ch)
                continue
            out.append(ch)
        while stack:
            out.append(stack.pop())
        return "".join(out)

    @staticmethod
    def _extract_json_block(text: str) -> str:
        cleaned = PRDOrchestrator._normalize_json_candidate(text)

        # Try to extract the first balanced JSON object to avoid malformed tail content.
        start = cleaned.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return cleaned[start : i + 1]

        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        candidate = PRDOrchestrator._normalize_json_candidate(text)
        start = candidate.find("{")
        if start >= 0:
            candidate = candidate[start:]
        # Remove trailing commas before closing tokens.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        # If a string is left open at the end, close it.
        in_string = False
        escaped = False
        for ch in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
        if in_string:
            if candidate.endswith("\\"):
                candidate += " "
            candidate += '"'

        brace_diff = candidate.count("{") - candidate.count("}")
        bracket_diff = candidate.count("[") - candidate.count("]")
        if bracket_diff > 0:
            candidate += "]" * bracket_diff
        if brace_diff > 0:
            candidate += "}" * brace_diff
        return candidate

    @staticmethod
    def _strip_json_comments(text: str) -> str:
        # Best-effort: remove JS-style comments that break JSON parsing.
        cleaned = re.sub(r"/\\*[\\s\\S]*?\\*/", "", text)
        cleaned = re.sub(r"(?m)^\\s*//.*$", "", cleaned)
        return cleaned

    @staticmethod
    def _quote_unquoted_keys(text: str) -> str:
        # Convert {foo: 1, bar_baz: "x"} -> {"foo": 1, "bar_baz": "x"}
        return re.sub(
            r'([\\{\\[,]\\s*)([A-Za-z_][A-Za-z0-9_]*)(\\s*):',
            r'\\1"\\2"\\3:',
            text,
        )

    @staticmethod
    def _maybe_convert_single_quotes(text: str) -> str:
        # If output is JSON-ish but uses lots of single quotes, convert to double quotes.
        # This is intentionally conservative to avoid breaking valid JSON.
        s = text
        if '"' in s:
            return s
        if s.count("'") < 4:
            return s

        def _repl(match: re.Match) -> str:
            inner = match.group(1)
            inner = inner.replace('\\"', '"').replace('"', '\\"')
            return f'"{inner}"'

        return re.sub(r"'([^'\\\\]*(?:\\\\.[^'\\\\]*)*)'", _repl, s)

    @staticmethod
    def _escape_newlines_in_strings(text: str) -> str:
        # JSON does not allow literal newlines inside quoted strings. Some models emit them anyway.
        out: list[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                    out.append(ch)
                    continue
                if ch == "\\":
                    escaped = True
                    out.append(ch)
                    continue
                if ch == '"':
                    in_string = False
                    out.append(ch)
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _escape_unescaped_quotes_in_strings(text: str) -> str:
        """
        Best-effort repair: models sometimes emit unescaped double-quotes inside JSON string
        values, which breaks parsing (often showing up as 'unterminated string literal').
        Heuristic: if we are inside a string and see a `"`, treat it as a closing quote only
        when the next non-whitespace character is one of `: , } ]` (key/value boundary).
        Otherwise, escape it as `\\"` and keep the string open.
        """
        out: list[str] = []
        in_string = False
        escaped = False
        n = len(text)
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                    out.append(ch)
                    continue
                if ch == "\\":
                    escaped = True
                    out.append(ch)
                    continue
                if ch == '"':
                    j = i + 1
                    while j < n and text[j] in " \t\r\n":
                        j += 1
                    nxt = text[j] if j < n else ""
                    if nxt in {":", ",", "}", "]"}:
                        in_string = False
                        out.append(ch)
                        continue
                    out.append('\\"')
                    continue
                out.append(ch)
                continue

            if ch == '"':
                in_string = True
                out.append(ch)
                continue
            out.append(ch)
        return "".join(out)

    @classmethod
    def _coerce_jsonish(cls, text: str) -> str:
        cleaned = cls._normalize_json_candidate(text)
        cleaned = cleaned.lstrip("\ufeff").strip()
        cleaned = cls._strip_json_comments(cleaned)
        cleaned = cls._quote_unquoted_keys(cleaned)
        cleaned = re.sub(r",\\s*([}\\]])", r"\\1", cleaned)
        cleaned = cls._maybe_convert_single_quotes(cleaned)
        cleaned = cls._escape_newlines_in_strings(cleaned)
        cleaned = cls._escape_unescaped_quotes_in_strings(cleaned)
        return cleaned.strip()

    @classmethod
    def _parse_structured(cls, raw: str, schema: type[BaseModel]) -> BaseModel:
        candidate = cls._extract_json_block(raw)
        parsed: Any
        last_exc: Exception | None = None
        try:
            parsed = json.loads(candidate)
        except Exception as exc0:
            last_exc = exc0
            try:
                coerced = cls._coerce_jsonish(candidate)
                if coerced != candidate:
                    parsed = json.loads(coerced)
                else:
                    raise ValueError("coerce skipped")
            except Exception as exc1:
                last_exc = exc1
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception as exc2:
                    last_exc = exc2
                    repaired = cls._repair_truncated_json(candidate)
                    balanced = cls._balance_json_like(repaired)
                    try:
                        parsed = json.loads(repaired)
                    except Exception as exc3:
                        last_exc = exc3
                        try:
                            parsed = ast.literal_eval(repaired)
                        except Exception as exc:
                            last_exc = exc
                            try:
                                parsed = json.loads(cls._coerce_jsonish(balanced))
                            except Exception as exc4:
                                last_exc = exc4
                                try:
                                    parsed = ast.literal_eval(balanced)
                                except Exception as exc5:
                                    last_exc = exc5
                                    try:
                                        parsed = cls._parse_loose_by_schema(balanced, schema)
                                    except Exception:
                                        # Final fallback: for single-field outputs, salvage the value instead of failing the whole run.
                                        try:
                                            fields = list(schema.model_fields.keys())
                                            if len(fields) == 1:
                                                key = fields[0]
                                                ann = schema.model_fields[key].annotation
                                                is_list = get_origin(ann) is list
                                                text = cls._normalize_json_candidate(candidate)
                                                # If the key is present, prefer extracting its value.
                                                m = re.search(rf'(?is)(?<![A-Za-z0-9_])"?{re.escape(key)}"?\s*:\s*(.*)$', text)
                                                raw_value = (m.group(1) if m else text).strip()
                                                raw_value = raw_value.rstrip(",").strip()
                                                value = cls._parse_loose_value(raw_value, schema, key)
                                                if not is_list and not isinstance(value, str):
                                                    value = str(value)
                                                return schema.model_validate({key: value})
                                        except Exception:
                                            pass
                                        raise ValueError(f"Unable to parse JSON output: {exc}") from last_exc

        if not isinstance(parsed, dict):
            raise ValueError("Structured output must be a JSON object")
        parsed = cls._sanitize_obj(parsed)
        parsed = cls._coerce_schema_lists(parsed, schema)
        parsed = cls._fill_missing_schema_fields(parsed, schema)
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise ValueError(str(exc))

    @classmethod
    def _parse_loose_by_schema(cls, text: str, schema: type[BaseModel]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        fields = list(schema.model_fields.keys())
        positions: list[tuple[str, int, int]] = []
        for key in fields:
            m = re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(key)}"?\s*:', text)
            if m:
                positions.append((key, m.start(), m.end()))
        if not positions:
            raise ValueError("No schema keys found in model output")
        positions.sort(key=lambda x: x[1])

        for idx, (key, _, end_pos) in enumerate(positions):
            next_start = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
            raw_value = text[end_pos:next_start].strip()
            raw_value = raw_value.rstrip(",").strip()
            payload[key] = cls._parse_loose_value(raw_value, schema, key)
        return payload

    @classmethod
    def _parse_loose_value(cls, raw_value: str, schema: type[BaseModel], key: str) -> Any:
        if not raw_value:
            return "Insufficient information provided."

        annotation = schema.model_fields[key].annotation
        is_list = get_origin(annotation) is list

        value = raw_value.strip()
        # Remove trailing braces that may belong to the outer object.
        while value.endswith("}") and not value.startswith("{"):
            value = value[:-1].rstrip()

        if value.startswith('"'):
            # Best-effort string extraction even if closing quote is missing.
            if len(value) >= 2 and value.endswith('"'):
                inner = value[1:-1]
            else:
                inner = value[1:]
            inner = inner.replace('\\"', '"').replace("\\n", "\n").replace("\\t", " ")
            return cls._split_to_list(inner) if is_list else inner

        if value.startswith("["):
            repaired = cls._repair_truncated_json(value)
            try:
                parsed = json.loads(repaired)
            except Exception:
                try:
                    parsed = ast.literal_eval(repaired)
                except Exception:
                    parsed = cls._split_to_list(value)
            if is_list:
                return parsed if isinstance(parsed, list) else cls._split_to_list(str(parsed))
            if isinstance(parsed, list):
                return ", ".join([str(v) for v in parsed])
            return str(parsed)

        if value.startswith("{"):
            repaired = cls._repair_truncated_json(value)
            try:
                parsed = json.loads(repaired)
            except Exception:
                try:
                    parsed = ast.literal_eval(repaired)
                except Exception:
                    parsed = value
            if is_list and isinstance(parsed, dict):
                return [f"{k}: {v}" for k, v in parsed.items()]
            return parsed if not is_list else cls._split_to_list(str(parsed))

        return cls._split_to_list(value) if is_list else value

    def _provider_config(self) -> tuple[str, str, str | None, str]:
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

    @staticmethod
    def _normalize_openai_base_url(base_url: str | None, provider: str) -> str | None:
        if not base_url:
            return base_url
        normalized = base_url.rstrip("/")
        if provider == "lmstudio":
            # LM Studio OpenAI-compatible endpoint is /v1/chat/completions.
            # Accept root, /api/v1, or /v1 and normalize to /v1.
            if normalized.endswith("/api/v1"):
                normalized = normalized[: -len("/api/v1")] + "/v1"
            elif not normalized.endswith("/v1"):
                normalized = normalized + "/v1"
        return normalized

    async def _invoke_llm(self, prompt: str, output_tokens: int) -> tuple[str, int, str]:
        model_name, api_key, base_url, provider = self._provider_config()
        if not api_key:
            raise ValueError("LLM API key not configured")

        normalized_base_url = self._normalize_openai_base_url(base_url, provider)
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
        tokens_used = int(usage.get("total_tokens") or 0) or self._estimate_tokens(prompt + "\n" + text)
        return text, max(tokens_used, 1), model_name

    def _bounded_payload(self, payload: dict[str, Any], *, max_tokens: int = MAX_STAGE_INPUT_TOKENS) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False)
        if self._estimate_tokens(raw) <= max_tokens:
            return payload

        compressed: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                compressed[key] = TokenLimiter.compress_text(value, max_tokens=250)
            elif isinstance(value, list):
                trimmed: list[Any] = []
                for item in value[:30]:
                    if isinstance(item, str):
                        trimmed.append(TokenLimiter.compress_text(item, max_tokens=40))
                    elif isinstance(item, dict):
                        trimmed.append({k: TokenLimiter.compress_text(str(v), max_tokens=25) for k, v in item.items()})
                    else:
                        trimmed.append(item)
                compressed[key] = trimmed
            else:
                compressed[key] = value

        compressed_raw = json.dumps(compressed, ensure_ascii=False)
        if self._estimate_tokens(compressed_raw) > max_tokens:
            # As a last resort, collapse into a short summary blob.
            compressed = {"summary": TokenLimiter.compress_text(raw, max_tokens=max(120, int(max_tokens * 0.7)))}
        return compressed

    async def _run_stage(
        self,
        stage_name: str,
        schema: type[BaseModel],
        stage_instruction: str,
        payload: dict[str, Any],
        *,
        max_output_tokens: int | None = None,
        partial_markdown: str | None = None,
        clarification_questions: dict[str, str] | None = None,
    ) -> StageRunResult:
        await self._check_control(stage_name)
        await self._emit_progress({"stage": stage_name, "status": "running"})

        # Keep prompt + completion under provider context window to avoid 400 "Context size exceeded".
        context_window = self._context_window_tokens()
        safety = self._context_safety_margin()
        max_total_tokens = max(512, context_window - safety)

        def build_prompt(source_payload: dict[str, Any]) -> tuple[str, int, int]:
            augmented_task = stage_instruction
            if clarification_questions:
                augmented_task = (
                    f"{stage_instruction} If user_provided_clarifications is present in input, treat it as ground truth "
                    "and do not output 'Insufficient information provided.' for those fields."
                )

            prompt_prefix = (
                f"System: {SYSTEM_CONTENT_ONLY}\n"
                f"Style: {DOC_STYLE_GUIDE}\n"
                f"Depth: {DETAIL_DEPTH_GUIDE}\n"
                f"Stage: {stage_name}\n"
                f"Task: {augmented_task}\n"
                "Rules: output a single valid JSON object only. Start with '{' and end with '}'. No prose, markdown, or code fences.\n"
                "Rules: all string values must be single-line. Use \\n inside strings for line breaks.\n"
                f"Allowed keys: {', '.join(schema.model_fields.keys())}\n\n"
                "Input:\n"
            )
            prefix_tokens = self._estimate_tokens(prompt_prefix)

            desired_output_tokens = MAX_STAGE_OUTPUT_TOKENS
            if isinstance(max_output_tokens, int) and max_output_tokens > 0:
                desired_output_tokens = min(desired_output_tokens, max_output_tokens)
            min_input_tokens = 200
            available_for_json = max_total_tokens - prefix_tokens - desired_output_tokens
            if available_for_json < min_input_tokens:
                desired_output_tokens = max(256, max_total_tokens - prefix_tokens - min_input_tokens)
                available_for_json = max_total_tokens - prefix_tokens - desired_output_tokens
            max_input_tokens = max(min_input_tokens, min(MAX_STAGE_INPUT_TOKENS, available_for_json))

            source = self._bounded_payload(source_payload, max_tokens=max_input_tokens)
            input_json = json.dumps(source, ensure_ascii=False, indent=2)
            prompt = prompt_prefix + input_json

            for _ in range(4):
                prompt_tokens = self._estimate_tokens(prompt)
                if prompt_tokens + desired_output_tokens <= max_total_tokens:
                    break
                overflow = (prompt_tokens + desired_output_tokens) - max_total_tokens
                max_input_tokens = max(min_input_tokens, max_input_tokens - max(50, overflow))
                source = self._bounded_payload(source, max_tokens=max_input_tokens)
                input_json = json.dumps(source, ensure_ascii=False, indent=2)
                prompt = prompt_prefix + input_json

            input_tokens = self._estimate_tokens(prompt)
            if input_tokens + desired_output_tokens > max_total_tokens:
                raise ValueError(
                    f"Prompt too large for model context (prompt={input_tokens}, completion={desired_output_tokens}, "
                    f"limit={context_window}). Reduce input or increase DV_LLM_CONTEXT_WINDOW_TOKENS."
                )
            return prompt, input_tokens, desired_output_tokens

        retry_count = 0
        last_error: Exception | None = None

        clarification_rounds = 0
        stage_payload = dict(payload)
        prompt, input_tokens, desired_output_tokens = build_prompt(stage_payload)

        for _attempt in range(3):
            await self._check_control(stage_name)
            try:
                raw_text, used_tokens, model_name = await self._invoke_llm(prompt, desired_output_tokens)
                self.total_tokens_used += used_tokens
            except Exception as exc:
                # `_invoke_llm` failures used to bypass the parsing try/except below, leaving last_error as None.
                # That produced confusing failures like: "stage_X failed: None".
                last_error = exc
                retry_count = 1
                logger.warning(
                    "prd_stage_llm_invoke_failed stage=%s attempt=%s error=%s",
                    stage_name,
                    _attempt + 1,
                    exc,
                )
                continue
            try:
                parsed = self._parse_structured(raw_text, schema)
                # If the user already provided clarifications for this stage, apply them deterministically.
                # This avoids clarification loops where the model keeps outputting "Insufficient information provided."
                # even though answers are available.
                try:
                    clarifications = stage_payload.get("user_provided_clarifications") if isinstance(stage_payload, dict) else None
                    if isinstance(clarifications, dict) and clarifications:
                        patched = parsed.model_dump()
                        changed = False
                        for k, v in clarifications.items():
                            key = str(k or "").strip()
                            if key not in schema.model_fields:
                                continue
                            if not isinstance(v, str) or not v.strip():
                                continue
                            if self._is_insufficient(patched.get(key)):
                                patched[key] = v.strip()
                                changed = True
                        if changed:
                            parsed = schema.model_validate(patched)
                except Exception:
                    pass

                # If the model reported insufficient info, pause and ask the user, then re-run the stage with answers.
                missing_fields = self._missing_fields(parsed)
                if stage_name == "stage_16_features_user_stories":
                    # Section 6.1.2-6.1.5 is driven by `feature_details`. If it's missing or incomplete,
                    # ask for clarification (single question) instead of producing lots of "Insufficient..." blocks.
                    try:
                        details = getattr(parsed, "feature_details", None)
                        needs_details = not isinstance(details, list) or len(details) < 2
                        if not needs_details:
                            for d in details[:2]:
                                if not isinstance(d, FeatureDetail):
                                    needs_details = True
                                    break
                                if any(
                                    self._is_insufficient(getattr(d, attr, None))
                                    for attr in (
                                        "feature_name",
                                        "detailed_flows",
                                        "io_validation_rules",
                                        "error_failure_handling",
                                        "priority",
                                    )
                                ):
                                    needs_details = True
                                    break
                        if needs_details and "feature_details" not in missing_fields:
                            missing_fields.append("feature_details")

                        # Only ask for user_stories if the list shape itself is missing; otherwise the model
                        # should be able to generate reasonable stories without blocking the run.
                        stories = getattr(parsed, "user_stories", None)
                        needs_stories = not isinstance(stories, list) or len(stories) < 12
                        if needs_stories and "user_stories" not in missing_fields:
                            missing_fields.append("user_stories")
                    except Exception:
                        if "feature_details" not in missing_fields:
                            missing_fields.append("feature_details")
                if missing_fields and clarification_questions:
                    # If answers already exist in the run controls (common on retries / resume),
                    # use them immediately instead of re-asking the same question.
                    try:
                        snapshot = await self._control_snapshot()
                        answers_raw = snapshot.get("clarification_answers") if isinstance(snapshot, dict) else {}
                        answers_raw = answers_raw if isinstance(answers_raw, dict) else {}
                        already: dict[str, str] = {}
                        for f in missing_fields:
                            key = f"{stage_name}:{f}"
                            val = answers_raw.get(key) or answers_raw.get(f)
                            if isinstance(val, str) and val.strip():
                                already[f] = val.strip()
                        if already and all(f in already for f in missing_fields):
                            # If we keep landing here and the model still refuses to fill fields, we should
                            # fail with a useful error rather than "Unknown stage failure" after retries exhaust.
                            last_error = ValueError(
                                f"Model output still missing required fields after applying saved clarifications: {missing_fields}"
                            )
                            stage_payload = {**payload, "user_provided_clarifications": already}
                            prompt, input_tokens, desired_output_tokens = build_prompt(stage_payload)
                            retry_count = 1
                            continue
                    except Exception:
                        pass

                    clarification_rounds += 1
                    max_rounds = 4 if stage_name == "stage_16_features_user_stories" else 2
                    if clarification_rounds > max_rounds:
                        raise ValueError(f"Clarification loop exceeded for {stage_name}: {missing_fields}")

                    # One-by-one clarification: ask for only one missing field at a time.
                    if stage_name == "stage_16_features_user_stories":
                        if "feature_details" in missing_fields:
                            missing_fields = ["feature_details"]
                        else:
                            missing_fields = missing_fields[:1]
                    else:
                        missing_fields = missing_fields[:1]

                    answer_keys = [f"{stage_name}:{f}" for f in missing_fields]
                    questions = [clarification_questions.get(f, f"Please provide more information for: {f}") for f in missing_fields]
                    await self._emit_progress(
                        {
                            "stage": stage_name,
                            "status": "clarification_required",
                            "missing_fields": missing_fields,
                            "questions": questions,
                            "answer_keys": answer_keys,
                            "stage_output": self._progress_payload_from_model(parsed),
                            **({"partial_markdown": partial_markdown} if partial_markdown else {}),
                        }
                    )

                    # Wait for the user to respond (pause is set in DB by the API layer).
                    await self._check_control(stage_name)

                    snapshot = await self._control_snapshot()
                    answers_raw = snapshot.get("clarification_answers") if isinstance(snapshot, dict) else {}
                    answers_raw = answers_raw if isinstance(answers_raw, dict) else {}
                    stage_answers: dict[str, str] = {}
                    for f in missing_fields:
                        key = f"{stage_name}:{f}"
                        val = answers_raw.get(key) or answers_raw.get(f)
                        if isinstance(val, str) and val.strip():
                            stage_answers[f] = val.strip()
                    if not stage_answers:
                        last_error = ValueError(f"Clarification answers not found for: {missing_fields}")
                    stage_payload = {**payload, "user_provided_clarifications": stage_answers}
                    prompt, input_tokens, desired_output_tokens = build_prompt(stage_payload)
                    retry_count = 1
                    continue

                output_tokens = max(1, used_tokens - input_tokens)
                await log_llm_usage(
                    feature=f"prd_multistep:{stage_name}",
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tenant_id=self.tenant_id,
                    stage_name=stage_name,
                    retry_count=retry_count,
                )
                await self._emit_progress(
                    {
                        "stage": stage_name,
                        "status": "completed",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "retry_count": retry_count,
                        "stage_output": self._progress_payload_from_model(parsed),
                        "stage_output_full": parsed.model_dump(),
                        **({"partial_markdown": partial_markdown} if partial_markdown else {}),
                    }
                )
                return StageRunResult(output=parsed, input_tokens=input_tokens, output_tokens=output_tokens, retry_count=retry_count)
            except Exception as exc:
                last_error = exc
                retry_count = 1
                logger.warning("prd_stage_validation_failed stage=%s attempt=%s error=%s", stage_name, _attempt + 1, exc)

        # Stage-specific fallback: Stage 16 is a large structured section that can be flaky with some models.
        # If the model refuses to produce valid/complete objects even after clarification attempts,
        # synthesize a reasonable baseline from earlier stages so the run can continue.
        if stage_name == "stage_16_features_user_stories" and schema is Stage12UserStoriesOutput:
            try:
                fallback = self._fallback_stage16(stage_payload)
                await self._emit_progress(
                    {
                        "stage": stage_name,
                        "status": "completed",
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "retry_count": retry_count,
                        "stage_output": self._progress_payload_from_model(fallback),
                        "stage_output_full": fallback.model_dump(),
                        "fallback": True,
                        "fallback_reason": str(last_error) if last_error else "unknown",
                        **({"partial_markdown": partial_markdown} if partial_markdown else {}),
                    }
                )
                return StageRunResult(output=fallback, input_tokens=input_tokens, output_tokens=0, retry_count=retry_count)
            except Exception:
                # If fallback fails, proceed with the normal failure path below.
                pass

        if last_error is None:
            last_error = ValueError("Unknown stage failure")
        await self._emit_progress({"stage": stage_name, "status": "failed", "error": str(last_error)})
        raise ValueError(f"{stage_name} failed: {last_error}")

    def _fallback_stage16(self, payload: dict[str, Any]) -> Stage12UserStoriesOutput:
        features = payload.get("in_scope_features")
        if not isinstance(features, list):
            features = []
        features = [str(f).strip() for f in features if str(f).strip()]
        top1 = features[0] if len(features) >= 1 else "Core workflow"
        top2 = features[1] if len(features) >= 2 else "Insights and reporting"

        def feature_detail(name: str, priority: str) -> FeatureDetail:
            return FeatureDetail(
                feature_name=name,
                detailed_flows=(
                    f"- Happy path: user discovers {name}, configures it, and completes the primary task end-to-end.\n"
                    f"- Edge cases: empty state, invalid inputs, offline/slow network, duplicate submissions, permission denied.\n"
                    f"- State: draft, saved, updated, and deleted states are handled consistently."
                ),
                io_validation_rules=(
                    "- Validate required fields before submit.\n"
                    "- Enforce sane length limits; trim whitespace.\n"
                    "- Reject malformed payloads; return actionable errors.\n"
                    "- Ensure server-side validation matches client validation."
                ),
                error_failure_handling=(
                    "- Show inline errors for validation failures.\n"
                    "- Retry transient failures with backoff; avoid duplicate side effects.\n"
                    "- Provide clear fallback UI for offline mode.\n"
                    "- Log errors with correlation IDs for debugging."
                ),
                priority=priority,
            )

        user_story_ids = [
            "US-1.1",
            "US-1.2",
            "US-1.3",
            "US-2.1",
            "US-2.2",
            "US-2.3",
            "US-3.1",
            "US-3.2",
            "US-3.3",
            "US-4.1",
            "US-4.2",
            "US-4.3",
        ]
        descriptions = [
            f"As a user, I want to create and configure {top1} so that I can get started quickly.",
            f"As a user, I want to edit and manage items in {top1} so that my setup stays accurate over time.",
            f"As a user, I want to complete daily actions in {top1} so that I can track progress with minimal friction.",
            "As a user, I want reminders/notifications so that I do not forget important actions.",
            "As a user, I want to snooze or reschedule reminders so that interruptions are manageable.",
            "As a user, I want control over notification preferences so that messaging matches my needs.",
            f"As a user, I want to see progress analytics for {top2} so that I can understand trends.",
            "As a user, I want weekly/monthly summaries so that I can review outcomes over time.",
            "As a user, I want to export or share my data so that I can use it elsewhere.",
            "As an admin/user, I want to manage categories/tags so that content is organized.",
            "As a user, I want backup/sync so that my data is not lost across devices.",
            "As a user, I want to delete my account/data so that I can control my privacy.",
        ]

        def criteria(base: str) -> list[str]:
            return [
                f"{base} works end-to-end on the happy path.",
                "Validation errors are shown inline with clear next steps.",
                "The action is idempotent or protected from double-submit.",
            ]

        stories: list[UserStory] = []
        for sid, desc in zip(user_story_ids, descriptions):
            stories.append(UserStory(id=sid, description=desc, acceptance_criteria=criteria(sid)))

        return Stage12UserStoriesOutput(
            user_stories=stories,
            feature_details=[feature_detail(top1, "P0"), feature_detail(top2, "P1")],
        )

    async def generate_content(self, payload: dict[str, Any]) -> PRDContent:
        today = datetime.now(timezone.utc).strftime("%B %d, %Y")

        async def emit_partial(stage: str, parsed: BaseModel, partial_markdown: str) -> None:
            # `_run_stage` already emitted "completed" once; this emits a follow-up event that includes `partial_markdown`.
            try:
                await self._emit_progress(
                    {
                        "stage": stage,
                        "status": "completed",
                        "stage_output": self._progress_payload_from_model(parsed),
                        "partial_markdown": partial_markdown,
                    }
                )
            except Exception:
                pass

        LOADING = "<!-- dv:loading -->"
        # Maintain a PRDContent-shaped state so we can re-render a coherent partial markdown after every stage.
        prd_state: dict[str, Any] = {
            "executive_summary": LOADING,
            "core_problem": LOADING,
            "why_tools_fail": LOADING,
            "success_meaning": LOADING,
            "primary_objective": LOADING,
            "success_metrics": [LOADING],
            "leading_indicators": [LOADING],
            "personas": [],
            "in_scope_features": [LOADING],
            "out_of_scope": [LOADING],
            "user_stories": [],
            "feature_details": [],
            "architecture_summary": LOADING,
            "data_model_summary": LOADING,
            "api_summary": LOADING,
            "slack_integration_summary": LOADING,
            "security_summary": LOADING,
            "ui_summary": LOADING,
            "copy_localization_notes": LOADING,
            "dependencies_summary": LOADING,
            "non_functional_summary": LOADING,
            "testing_summary": LOADING,
            "launch_plan_summary": LOADING,
            "target_release_dates": LOADING,
            "open_questions": [LOADING],
            "assumptions": [LOADING],
            "risks": [LOADING],
            "definition_of_done": [LOADING],
            "glossary": [LOADING],
        }

        def render_partial() -> str:
            # Render using the same hierarchical renderer as final output, but with LOADING placeholders.
            return render_hierarchical_prd(
                PRDContent(**prd_state),
                today,
                project_title=str(payload.get("title") or "").strip() or None,
                project_code=self.project_id,
            )

        resume_idx = None
        if self._resume_from_stage and self._resume_from_stage in PRD_STAGE_ORDER:
            resume_idx = PRD_STAGE_ORDER.index(self._resume_from_stage)

        def _should_use_cache(stage_name: str) -> bool:
            if resume_idx is None:
                return False
            if stage_name not in PRD_STAGE_ORDER:
                return False
            return PRD_STAGE_ORDER.index(stage_name) < resume_idx

        def _load_cached(stage_name: str, schema: type[BaseModel]) -> BaseModel | None:
            if not _should_use_cache(stage_name):
                return None
            raw = self._stage_cache.get(stage_name)
            if not isinstance(raw, dict) or not raw:
                return None
            try:
                return schema.model_validate(raw)
            except Exception:
                return None

        if self.project_id:
            try:
                await sync_project_knowledge_chunks(tenant_id=self.tenant_id, project_id=self.project_id)
                retrieval_query = "\n".join(
                    [
                        f"title: {payload.get('title')}",
                        f"problem_statement: {payload.get('problem_statement')}",
                        "target_users: "
                        + (
                            ", ".join(str(x) for x in (payload.get("target_users") or []))
                            if isinstance(payload.get("target_users"), list)
                            else str(payload.get("target_users") or "")
                        ),
                        "features: " + ", ".join(str(x) for x in (payload.get("features") or [])),
                        f"additional_notes: {payload.get('additional_notes')}",
                    ]
                )
                self.retrieved_chunks = await retrieve_project_knowledge_chunks(
                    tenant_id=self.tenant_id,
                    project_id=self.project_id,
                    query_text=retrieval_query,
                    top_k=6,
                )
            except Exception:
                self.retrieved_chunks = []

        cached01 = _load_cached("stage_01_context_snapshot", Stage01ContextOutput)
        if cached01:
            stage01 = StageRunResult(output=cached01, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage01 = await self._run_stage(
                stage_name="stage_01_context_snapshot",
                schema=Stage01ContextOutput,
                stage_instruction=(
                    "Fill only context_summary. Summarize the intake into 2-4 paragraphs of factual context and constraints. "
                    "Do not invent data, metrics, integrations, or technologies."
                ),
                payload={
                    "title": payload["title"],
                    "problem_statement": payload["problem_statement"],
                    "target_users": payload["target_users"],
                    "features": payload["features"],
                    "additional_notes": payload.get("additional_notes", ""),
                    "retrieved_project_knowledge_chunks": self.retrieved_chunks,
                },
                max_output_tokens=500,
            )
            await emit_partial("stage_01_context_snapshot", stage01.output, render_partial())

        cached02 = _load_cached("stage_02_intro_summary", Stage02IntroSummaryOutput)
        if cached02:
            stage02 = StageRunResult(output=cached02, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage02 = await self._run_stage(
                stage_name="stage_02_intro_summary",
                schema=Stage02IntroSummaryOutput,
                stage_instruction=(
                    "Fill executive_summary and success_meaning. "
                    "Executive summary: 2-4 detailed paragraphs, enterprise PRD tone. "
                    "Success meaning: 1-2 paragraphs describing outcomes for users and the business."
                ),
                payload={
                    **payload,
                    "context_summary": stage01.output.context_summary,
                    "retrieved_project_knowledge_chunks": self.retrieved_chunks,
                },
                max_output_tokens=900,
                clarification_questions={
                    "executive_summary": "Please provide a detailed executive summary (what it is, who it's for, and why it matters).",
                    "success_meaning": "What does success look like after launch? Describe outcomes for users and the business.",
                },
            )
        prd_state["executive_summary"] = stage02.output.executive_summary
        prd_state["success_meaning"] = stage02.output.success_meaning
        if not cached02:
            await emit_partial("stage_02_intro_summary", stage02.output, render_partial())

        cached03 = _load_cached("stage_03_doc_metadata", Stage03DocMetadataOutput)
        if cached03:
            stage03 = StageRunResult(output=cached03, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage03 = await self._run_stage(
                stage_name="stage_03_doc_metadata",
                schema=Stage03DocMetadataOutput,
                stage_instruction=(
                    "Fill doc_version, doc_author, doc_reviewers, doc_approvers, applicable_teams. "
                    "If unknown, infer only from provided org/team context; otherwise ask for clarification."
                ),
                payload={
                    **payload,
                    "context_summary": stage01.output.context_summary,
                    "retrieved_project_knowledge_chunks": self.retrieved_chunks,
                },
                max_output_tokens=350,
                clarification_questions={
                    "doc_author": "Who is the author/owner of this PRD (name or team)?",
                    "doc_reviewers": "Who are the reviewers (teams or names)?",
                    "doc_approvers": "Who are the approvers (teams or names)?",
                    "applicable_teams": "Which teams are applicable (Product/Eng/Design/QA/DevOps/Compliance etc.)?",
                },
            )
        prd_state["doc_version"] = stage03.output.doc_version or "1.0"
        prd_state["doc_author"] = stage03.output.doc_author or "Product Team"
        prd_state["doc_reviewers"] = stage03.output.doc_reviewers or ["Engineering", "Design", "QA"]
        prd_state["doc_approvers"] = stage03.output.doc_approvers or ["Product Leadership"]
        prd_state["applicable_teams"] = stage03.output.applicable_teams or ["Product", "Engineering", "Design", "QA", "DevOps", "Compliance"]
        if not cached03:
            await emit_partial("stage_03_doc_metadata", stage03.output, render_partial())

        cached04 = _load_cached("stage_04_intro_problem_context", Stage03CoreProblemOutput)
        if cached04:
            stage04 = StageRunResult(output=cached04, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage04 = await self._run_stage(
                stage_name="stage_04_intro_problem_context",
                schema=Stage03CoreProblemOutput,
                stage_instruction="Fill only core_problem. Be specific about the pain and current workflow breakdowns.",
                payload={
                    **payload,
                    "context_summary": stage01.output.context_summary,
                    "executive_summary": stage02.output.executive_summary,
                    "retrieved_project_knowledge_chunks": self.retrieved_chunks,
                },
                max_output_tokens=700,
                clarification_questions={
                    "core_problem": "What is the core problem you want to solve? Share concrete examples of the current pain.",
                },
            )
        prd_state["core_problem"] = stage04.output.core_problem
        if not cached04:
            await emit_partial("stage_04_intro_problem_context", stage04.output, render_partial())

        cached05 = _load_cached("stage_05_intro_why_build", Stage04WhyToolsFailOutput)
        if cached05:
            stage05 = StageRunResult(output=cached05, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage05 = await self._run_stage(
                stage_name="stage_05_intro_why_build",
                schema=Stage04WhyToolsFailOutput,
                stage_instruction="Fill only why_tools_fail. List concrete failure modes and why current tools/processes break down.",
                payload={
                    **payload,
                    "context_summary": stage01.output.context_summary,
                    "core_problem": stage04.output.core_problem,
                    "retrieved_project_knowledge_chunks": self.retrieved_chunks,
                },
                max_output_tokens=700,
                clarification_questions={
                    "why_tools_fail": "Why do existing tools/processes fail today? List the specific gaps and failure modes.",
                },
            )
        prd_state["why_tools_fail"] = stage05.output.why_tools_fail
        if not cached05:
            await emit_partial("stage_05_intro_why_build", stage05.output, render_partial())

        cached06 = _load_cached("stage_06_intro_strategy_okrs", Stage05StrategyOkrsOutput)
        if cached06:
            stage06 = StageRunResult(output=cached06, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage06 = await self._run_stage(
            stage_name="stage_06_intro_strategy_okrs",
            schema=Stage05StrategyOkrsOutput,
            stage_instruction="Fill only strategy_okrs. Reference strategy/roadmap/OKRs in plain language.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=450,
            clarification_questions={
                "strategy_okrs": "Share any strategy/roadmap/OKRs this project maps to (or say 'none').",
            },
            )
        prd_state["strategy_okrs"] = stage06.output.strategy_okrs
        if not cached06:
            await emit_partial("stage_06_intro_strategy_okrs", stage06.output, render_partial())

        cached07 = _load_cached("stage_07_goals_primary", Stage06PrimaryObjectiveOutput)
        if cached07:
            stage07 = StageRunResult(output=cached07, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage07 = await self._run_stage(
            stage_name="stage_07_goals_primary",
            schema=Stage06PrimaryObjectiveOutput,
            stage_instruction="Fill only primary_objective. One crisp paragraph.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "success_meaning": stage02.output.success_meaning,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=350,
            clarification_questions={
                "primary_objective": "What is the primary objective for this release (one crisp goal)?",
            },
            )
        prd_state["primary_objective"] = stage07.output.primary_objective
        if not cached07:
            await emit_partial("stage_07_goals_primary", stage07.output, render_partial())

        cached08 = _load_cached("stage_08_goals_kpis", Stage08KpisOutput)
        if cached08:
            stage08 = StageRunResult(output=cached08, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage08 = await self._run_stage(
            stage_name="stage_08_goals_kpis",
            schema=Stage08KpisOutput,
            stage_instruction="Fill success_metrics and leading_indicators. Provide measurable bullet items.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "primary_objective": stage07.output.primary_objective,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=650,
            clarification_questions={
                "success_metrics": "What are the key success metrics (KPIs) for this product/release? Provide measurable items.",
                "leading_indicators": "What leading indicators should we track to predict success early (before KPIs move)?",
            },
            )
        prd_state["success_metrics"] = stage08.output.success_metrics or [LOADING]
        prd_state["leading_indicators"] = stage08.output.leading_indicators or [LOADING]
        if not cached08:
            await emit_partial("stage_08_goals_kpis", stage08.output, render_partial())

        cached09 = _load_cached("stage_09_goals_done_criteria", Stage09DefinitionOfDoneOutput)
        if cached09:
            stage09 = StageRunResult(output=cached09, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage09 = await self._run_stage(
            stage_name="stage_09_goals_done_criteria",
            schema=Stage09DefinitionOfDoneOutput,
            stage_instruction="Fill only definition_of_done. 6-12 concrete go-live criteria items.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "success_metrics": stage08.output.success_metrics,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=450,
            clarification_questions={
                "definition_of_done": "Define what 'done' means for go-live (acceptance/gate criteria).",
            },
            )
        prd_state["definition_of_done"] = stage09.output.definition_of_done or []
        if not cached09:
            await emit_partial("stage_09_goals_done_criteria", stage09.output, render_partial())

        cached10 = _load_cached("stage_10_users_types", Stage10UserTypesOutput)
        if cached10:
            stage10 = StageRunResult(output=cached10, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage10 = await self._run_stage(
            stage_name="stage_10_users_types",
            schema=Stage10UserTypesOutput,
            stage_instruction="Fill only user_types. List the main user types (end users, admins, ops, partners).",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "target_users": payload.get("target_users"),
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=300,
            clarification_questions={
                "user_types": "Who are the main user types (end users, admins, ops, partners)?",
            },
            )
        prd_state["user_types"] = stage10.output.user_types or []
        if not cached10:
            await emit_partial("stage_10_users_types", stage10.output, render_partial())

        cached11 = _load_cached("stage_11_users_personas", Stage09PersonasOutput)
        if cached11:
            stage11 = StageRunResult(output=cached11, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage11 = await self._run_stage(
            stage_name="stage_11_users_personas",
            schema=Stage09PersonasOutput,
            stage_instruction=(
                "Fill only personas. Create exactly 3 entries. "
                "Each persona must have name, role, description, pain_points, goals."
            ),
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "user_types": stage10.output.user_types,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=900,
            clarification_questions={
                "personas": "Who are the target personas? Provide 3 personas with role, goals, pain points, and context.",
            },
            )
        prd_state["personas"] = stage11.output.personas
        if not cached11:
            await emit_partial("stage_11_users_personas", stage11.output, render_partial())

        cached12 = _load_cached("stage_12_users_market_optional", Stage12MarketOverviewOutput)
        if cached12:
            stage12 = StageRunResult(output=cached12, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage12 = await self._run_stage(
            stage_name="stage_12_users_market_optional",
            schema=Stage12MarketOverviewOutput,
            stage_instruction="Fill only market_overview. Keep it short; if unknown, ask for clarification.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=350,
            clarification_questions={
                "market_overview": "Optional: what market/customer segment is this for (industry, size, region)?",
            },
            )
        prd_state["market_overview"] = stage12.output.market_overview
        if not cached12:
            await emit_partial("stage_12_users_market_optional", stage12.output, render_partial())

        cached13 = _load_cached("stage_13_scope_in", Stage10InScopeFeaturesOutput)
        if cached13:
            stage13 = StageRunResult(output=cached13, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage13 = await self._run_stage(
            stage_name="stage_13_scope_in",
            schema=Stage10InScopeFeaturesOutput,
            stage_instruction="Fill only in_scope_features. 8-15 items, concrete and testable.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=600,
            clarification_questions={
                "in_scope_features": "List the in-scope features for this release (8-15 concrete items).",
            },
            )
        prd_state["in_scope_features"] = stage13.output.in_scope_features or [LOADING]
        if not cached13:
            await emit_partial("stage_13_scope_in", stage13.output, render_partial())

        cached14 = _load_cached("stage_14_scope_out", Stage11OutOfScopeOutput)
        if cached14:
            stage14 = StageRunResult(output=cached14, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage14 = await self._run_stage(
            stage_name="stage_14_scope_out",
            schema=Stage11OutOfScopeOutput,
            stage_instruction="Fill only out_of_scope. 5-12 items, clear boundaries.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=500,
            clarification_questions={
                "out_of_scope": "List what is explicitly out-of-scope for this release (5-12 clear boundaries).",
            },
            )
        prd_state["out_of_scope"] = stage14.output.out_of_scope or [LOADING]
        if not cached14:
            await emit_partial("stage_14_scope_out", stage14.output, render_partial())

        cached15 = _load_cached("stage_15_scope_assumptions", Stage15AssumptionsOutput)
        if cached15:
            stage15 = StageRunResult(output=cached15, input_tokens=0, output_tokens=0, retry_count=0)
        else:
            stage15 = await self._run_stage(
            stage_name="stage_15_scope_assumptions",
            schema=Stage15AssumptionsOutput,
            stage_instruction="Fill only assumptions. Include assumptions about data, integrations, tech stack, timelines.",
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=550,
            clarification_questions={
                "assumptions": "List key assumptions about data, integrations, tech stack, and timelines.",
            },
            )
        prd_state["assumptions"] = stage15.output.assumptions or []
        if not cached15:
            await emit_partial("stage_15_scope_assumptions", stage15.output, render_partial())

        stage16 = await self._run_stage(
            stage_name="stage_16_features_user_stories",
            schema=Stage12UserStoriesOutput,
            stage_instruction=(
                "Fill user_stories and feature_details.\n"
                "user_stories: return exactly 12 user stories with IDs US-1.1..US-4.3. "
                "Each must include description and acceptance_criteria list with measurable statements.\n"
                "feature_details: return exactly 2 items describing the top 2 in-scope features. "
                "Each item must include feature_name, detailed_flows, io_validation_rules, error_failure_handling, and priority (P0/P1/P2)."
            ),
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "out_of_scope": stage14.output.out_of_scope,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=1100,
            clarification_questions={
                "user_stories": "Provide 12 user stories (US-1.1..US-4.3) with measurable acceptance criteria.",
                "feature_details": (
                    "For the top 2 in-scope features, provide: detailed flows (happy path + edge cases), "
                    "input/output + validation rules, error/failure handling, and priority (P0/P1/P2)."
                ),
            },
        )
        prd_state["user_stories"] = stage16.output.user_stories
        prd_state["feature_details"] = stage16.output.feature_details or []
        await emit_partial("stage_16_features_user_stories", stage16.output, render_partial())

        stage17 = await self._run_stage(
            stage_name="stage_17_architecture_data_api_integrations",
            schema=Stage17ArchitecturePackOutput,
            stage_instruction=(
                "Fill architecture_summary, data_model_summary, api_summary, slack_integration_summary, security_summary, dependencies_summary. "
                "Keep statements implementation-oriented and consistent with provided features only."
            ),
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "user_stories": [s.model_dump() for s in stage16.output.user_stories],
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=1500,
            clarification_questions={
                "architecture_summary": "Provide the high-level architecture approach (components, boundaries, and runtime behaviors).",
                "data_model_summary": "What are the core data entities and key business rules? Provide a high-level data model summary.",
                "api_summary": "Describe the key APIs needed (major route groups, auth, errors, patterns).",
                "slack_integration_summary": "If Slack (or any external integration) is needed, provide OAuth/events/permissions/failure handling details.",
                "security_summary": "What are the security requirements (auth, RBAC, encryption, PII handling, audit logs)?",
                "dependencies_summary": "List major integrations/dependencies and any constraints (licenses/SLAs).",
            },
        )
        prd_state["architecture_summary"] = stage17.output.architecture_summary
        prd_state["data_model_summary"] = stage17.output.data_model_summary
        prd_state["api_summary"] = stage17.output.api_summary
        prd_state["slack_integration_summary"] = stage17.output.slack_integration_summary
        prd_state["security_summary"] = stage17.output.security_summary
        prd_state["dependencies_summary"] = stage17.output.dependencies_summary
        await emit_partial("stage_17_architecture_data_api_integrations", stage17.output, render_partial())

        stage18 = await self._run_stage(
            stage_name="stage_18_ux_design",
            schema=Stage18UiSummaryOutput,
            stage_instruction=(
                "Fill ui_summary and copy_localization_notes. "
                "ui_summary: key flows, key screens, UX constraints, accessibility, design links if provided. "
                "copy_localization_notes: important copy guidelines, terminology, localization/internationalization requirements."
            ),
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "user_stories": [s.model_dump() for s in stage16.output.user_stories],
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=900,
            clarification_questions={
                "ui_summary": "Describe the key UX flows and screens, plus any UX constraints/accessibility needs.",
                "copy_localization_notes": "Any copy guidelines or localization requirements? (languages, terminology, tone, legal copy).",
            },
        )
        prd_state["ui_summary"] = stage18.output.ui_summary
        prd_state["copy_localization_notes"] = stage18.output.copy_localization_notes
        await emit_partial("stage_18_ux_design", stage18.output, render_partial())

        stage19 = await self._run_stage(
            stage_name="stage_19_delivery_quality",
            schema=Stage19FinalizeOutput,
            stage_instruction=(
                "Fill non_functional_summary, testing_summary, launch_plan_summary, ops_devops_summary, target_release_dates, open_questions, risks, glossary. "
                "Be practical and implementation-ready."
            ),
            payload={
                **payload,
                "context_summary": stage01.output.context_summary,
                "in_scope_features": stage13.output.in_scope_features,
                "out_of_scope": stage14.output.out_of_scope,
                "architecture_summary": stage17.output.architecture_summary,
                "security_summary": stage17.output.security_summary,
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
            max_output_tokens=1200,
            clarification_questions={
                "non_functional_summary": "Provide non-functional requirements (performance, reliability, scalability, observability, DR).",
                "testing_summary": "Provide a testing & QA plan (scenarios, automation focus, UAT needs).",
                "launch_plan_summary": "Provide a release plan and milestones (phased rollout, flags, rollback).",
                "ops_devops_summary": "Provide DevOps/ops plan (deployment strategy, environments, monitoring, backups/DR).",
                "target_release_dates": "What are the target release dates (high-level)? Provide tentative dates/timeframes if exact dates are unknown.",
                "open_questions": "List open questions / decisions pending that block clarity.",
                "risks": "List key risks and constraints.",
                "glossary": "Provide a short glossary of key terms.",
            },
        )
        prd_state["non_functional_summary"] = stage19.output.non_functional_summary
        prd_state["testing_summary"] = stage19.output.testing_summary
        prd_state["launch_plan_summary"] = stage19.output.launch_plan_summary
        prd_state["ops_devops_summary"] = stage19.output.ops_devops_summary
        prd_state["target_release_dates"] = stage19.output.target_release_dates
        prd_state["open_questions"] = stage19.output.open_questions or []
        prd_state["risks"] = stage19.output.risks or []
        prd_state["glossary"] = stage19.output.glossary or []
        await emit_partial("stage_19_delivery_quality", stage19.output, render_partial())

        return PRDContent(**prd_state)


def _bullets(items: list[str]) -> str:
    clean = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not clean:
        return "- Insufficient information provided."
    return "\n".join([f"- {item}" for item in clean])


def _split_sentences(text: str, max_items: int = 4) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return ["Insufficient information provided."]
    parts = [p.strip(" -") for p in re.split(r"[.;\n]+", text) if p.strip()]
    if not parts:
        return ["Insufficient information provided."]
    return parts[:max_items]


def _feature_slice(features: list[str], start: int, count: int) -> list[str]:
    sliced = [f.strip() for f in features[start : start + count] if isinstance(f, str) and f.strip()]
    return sliced or ["Insufficient information provided."]


def _map_personas(personas: list[Persona]) -> list[Persona]:
    defaults = [
        Persona(name="Sarah", role="Engineering Lead", description="Insufficient information provided."),
        Persona(name="David", role="Product Manager", description="Insufficient information provided."),
        Persona(name="Maya", role="Startup CTO/Founder", description="Insufficient information provided."),
    ]
    mapped = list(personas[:3])
    while len(mapped) < 3:
        mapped.append(defaults[len(mapped)])
    return mapped


def _story_map(stories: list[UserStory]) -> dict[str, UserStory]:
    ids = [
        "US-1.1", "US-1.2", "US-1.3", "US-2.1", "US-2.2", "US-2.3",
        "US-3.1", "US-3.2", "US-3.3", "US-4.1", "US-4.2", "US-4.3",
    ]
    source = {s.id.strip().upper(): s for s in stories if isinstance(s.id, str) and s.id.strip()}
    filler = iter(stories)
    for sid in ids:
        if sid in source:
            continue
        try:
            candidate = next(filler)
            source[sid] = UserStory(id=sid, description=candidate.description, acceptance_criteria=candidate.acceptance_criteria)
        except StopIteration:
            source[sid] = UserStory(id=sid, description="Insufficient information provided.", acceptance_criteria=[])
    return source


def render_hierarchical_prd(
    prd: PRDContent,
    today: str,
    *,
    project_title: str | None = None,
    project_code: str | None = None,
) -> str:
    personas = _map_personas(prd.personas)
    stories = _story_map(prd.user_stories)

    md = ""
    title = (project_title or "").strip() or "Insufficient information provided."
    code = (project_code or "").strip() or "Insufficient information provided."

    md += "# Product Requirements Document (PRD)\n\n"

    md += "## 1. Document header / metadata\n"
    md += f"### 1.1 Title and project code\n{title}\n\nProject code: {code}\n\n"
    md += "### 1.2 Version, author, reviewers, approvers\n"
    md += f"**Version:** {prd.doc_version or '1.0'}\n\n"
    md += f"**Author:** {prd.doc_author or 'Product Team'}\n\n"
    reviewers = prd.doc_reviewers or []
    md += f"**Reviewers:** {', '.join(reviewers) if reviewers else 'Insufficient information provided.'}\n\n"
    approvers = prd.doc_approvers or []
    md += f"**Approvers:** {', '.join(approvers) if approvers else 'Insufficient information provided.'}\n\n"
    md += "### 1.3 Date of creation and last update\n"
    md += f"**Created:** {today}\n\n"
    md += f"**Last updated:** {today}\n\n"
    md += "### 1.4 Applicable teams (Product, Eng, Design, QA, DevOps, Compliance, etc.)\n"
    md += _bullets(prd.applicable_teams or ["Insufficient information provided."]) + "\n\n"

    md += "## 2. Introduction & background\n"
    md += f"### 2.1 Problem statement / context\n{prd.core_problem}\n\n"
    md += "### 2.2 Why this project is being built (pain points, business need)\n"
    md += f"{prd.why_tools_fail}\n\n"
    md += "### 2.3 Link to strategy / roadmap / OKRs\n"
    md += f"{prd.strategy_okrs or 'Insufficient information provided.'}\n\n"

    md += "## 3. Goals & success metrics\n"
    md += f"### 3.1 Product goals (what \"good\" looks like)\n{prd.primary_objective}\n\n"
    md += "### 3.2 Key success metrics (KPIs)\n"
    md += _bullets(prd.success_metrics or ["Insufficient information provided."]) + "\n\n"
    md += "### 3.3 Go-live / success criteria (what \"done\" means)\n"
    md += _bullets(prd.definition_of_done or ["Insufficient information provided."]) + "\n\n"

    md += "## 4. Target users & personas\n"
    md += "### 4.1 Who are the main user types (end users, admins, ops, partners)\n"
    roles = [x.strip() for x in (prd.user_types or []) if isinstance(x, str) and x.strip()]
    if not roles:
        roles = [p.role.strip() for p in personas if isinstance(p.role, str) and p.role.strip()]
    if roles:
        md += _bullets(sorted(set(roles))) + "\n\n"
    else:
        md += "Insufficient information provided.\n\n"

    md += "### 4.2 User personas with roles, goals, and pain points\n"
    if personas:
        for p in personas:
            md += f"#### {p.name}\n"
            md += f"Role: {p.role}\n\n"
            md += f"{p.description}\n\n"
            md += f"Pain points:\n{_bullets(p.pain_points) if p.pain_points else '- Insufficient information provided.'}\n\n"
            md += f"Goals:\n{_bullets(p.goals) if p.goals else '- Insufficient information provided.'}\n\n"
    else:
        md += "Insufficient information provided.\n\n"

    md += "### 4.3 Market / customer segment overview (optional)\n"
    md += f"{prd.market_overview or 'Insufficient information provided.'}\n\n"

    md += "## 5. Scope and boundaries\n"
    md += "### 5.1 In-scope features (what this release will cover)\n"
    md += _bullets(prd.in_scope_features or ["Insufficient information provided."]) + "\n\n"
    md += "### 5.2 Out-of-scope items (what will be handled later or in another project)\n"
    md += _bullets(prd.out_of_scope or ["Insufficient information provided."]) + "\n\n"
    md += "### 5.3 Assumptions about data, integrations, tech stack, timelines\n"
    md += _bullets(prd.assumptions or ["Insufficient information provided."]) + "\n\n"

    md += "## 6. Features & functional requirements\n"
    feature_names = [x.strip() for x in (prd.in_scope_features or []) if isinstance(x, str) and x.strip()]
    while len(feature_names) < 2:
        feature_names.append(f"Feature {len(feature_names) + 1}")

    story_list = [s for s in prd.user_stories if isinstance(s, UserStory)]
    detail_list = [d for d in (prd.feature_details or []) if isinstance(d, FeatureDetail)]
    for idx, fname in enumerate(feature_names[:2], start=1):
        md += f"### 6.{idx} {fname}\n"
        md += f"#### 6.{idx}.1 User story / use case (as a user, I want to... so that...)\n"
        us = story_list[idx - 1] if len(story_list) >= idx else None
        fd = detail_list[idx - 1] if len(detail_list) >= idx else None
        if us:
            md += f"{us.description}\n\n"
        else:
            md += "Insufficient information provided.\n\n"

        md += f"#### 6.{idx}.2 Detailed behavior / flows (happy path + edge cases)\n"
        if fd and isinstance(fd.detailed_flows, str) and fd.detailed_flows.strip():
            md += f"{fd.detailed_flows}\n\n"
        else:
            md += "Insufficient information provided.\n\n"
        md += f"#### 6.{idx}.3 Input / output, validation rules\n"
        if fd and isinstance(fd.io_validation_rules, str) and fd.io_validation_rules.strip():
            md += f"{fd.io_validation_rules}\n\n"
        else:
            md += "Insufficient information provided.\n\n"
        md += f"#### 6.{idx}.4 Error / failure handling\n"
        if fd and isinstance(fd.error_failure_handling, str) and fd.error_failure_handling.strip():
            md += f"{fd.error_failure_handling}\n\n"
        else:
            md += "Insufficient information provided.\n\n"
        md += f"#### 6.{idx}.5 Priority (P0 / P1 / P2)\n"
        if fd and isinstance(fd.priority, str) and fd.priority.strip():
            md += f"{fd.priority}\n\n"
        else:
            md += "Insufficient information provided.\n\n"
        md += f"#### 6.{idx}.6 Acceptance criteria (what QA will test against)\n"
        if us and us.acceptance_criteria:
            md += _bullets(us.acceptance_criteria) + "\n\n"
        else:
            md += "Insufficient information provided.\n\n"

    md += "## 7. Non-functional requirements\n"
    md += f"### 7.1 Performance (latency, throughput, SLAs)\n{prd.non_functional_summary}\n\n"
    md += f"### 7.2 Reliability / availability (uptime, retry logic)\n{prd.non_functional_summary}\n\n"
    md += f"### 7.3 Security (auth, RBAC, data encryption, PII handling)\n{prd.security_summary}\n\n"
    md += f"### 7.4 Scalability, load limits, horizontal vs vertical\n{prd.non_functional_summary}\n\n"
    md += f"### 7.5 Compliance (GDPR, HIPAA, SOC2, internal policy)\n{prd.security_summary}\n\n"
    md += f"### 7.6 Observability (logging, monitoring, alerting)\n{prd.non_functional_summary}\n\n"
    md += f"### 7.7 Disaster recovery / backup strategy\n{prd.non_functional_summary}\n\n"

    md += "## 8. User experience and design\n"
    md += f"### 8.1 Key user flows / task flows\n{prd.ui_summary}\n\n"
    md += "### 8.2 Design links (Figma, Zeplin, sketches, mockups)\n"
    md += "Insufficient information provided.\n\n"
    md += f"### 8.3 UX guidelines / constraints (brand, accessibility, device support)\n{prd.ui_summary}\n\n"
    md += "### 8.4 Copy / localization notes (if applicable)\n"
    md += f"{prd.copy_localization_notes or 'Insufficient information provided.'}\n\n"

    md += "## 9. Integrations & dependencies\n"
    md += f"### 9.1 External systems (3rd-party APIs, partners, legacy systems)\n{prd.dependencies_summary}\n\n"
    md += f"### 9.2 Internal systems (CRM, ERP, auth service, data warehouse, etc.)\n{prd.api_summary}\n\n"
    md += f"### 9.3 Data dependencies (feeds, pipelines, CDC, batch vs real-time)\n{prd.data_model_summary}\n\n"
    md += f"### 9.4 Legal / contractual dependencies (licenses, SLAs, approvals)\n{prd.dependencies_summary}\n\n"

    md += "## 10. Data model & business rules\n"
    md += f"### 10.1 High-level entities / tables / schemas\n{prd.data_model_summary}\n\n"
    md += f"### 10.2 Core business rules\n{prd.data_model_summary}\n\n"
    md += f"### 10.3 Data retention / purge policies\n{prd.security_summary}\n\n"
    md += f"### 10.4 Reporting / analytics needs (dashboards, exports)\n{prd.success_meaning}\n\n"

    md += "## 11. Release plan & milestones\n"
    md += f"### 11.1 Release strategy (phased rollout, feature flags, A/B tests)\n{prd.launch_plan_summary}\n\n"
    md += f"### 11.2 Key milestones (design freeze, backend ready, UAT, production deploy)\n{prd.launch_plan_summary}\n\n"
    md += "### 11.3 Target release dates\n"
    md += f"{prd.target_release_dates or 'Insufficient information provided.'}\n\n"
    md += f"### 11.4 Rollback plan\n{prd.launch_plan_summary}\n\n"

    md += "## 12. Testing & QA\n"
    md += f"### 12.1 Test scope (manual / automated)\n{prd.testing_summary}\n\n"
    md += f"### 12.2 Key test scenarios linked to features\n{prd.testing_summary}\n\n"
    md += f"### 12.3 Non-functional test focus (performance, security, soak, chaos, etc.)\n{prd.testing_summary}\n\n"
    md += f"### 12.4 UAT / staging environment requirements\n{prd.testing_summary}\n\n"

    md += "## 13. Operations & DevOps\n"
    md += "### 13.1 Deployment strategy (canary, blue-green, CI/CD)\n"
    md += f"{prd.ops_devops_summary or 'Insufficient information provided.'}\n\n"
    md += "### 13.2 Environment requirements (dev, qa, staging, prod, data volumes)\n"
    md += f"{prd.ops_devops_summary or 'Insufficient information provided.'}\n\n"
    md += "### 13.3 Monitoring / alerting thresholds\n"
    md += f"{prd.ops_devops_summary or 'Insufficient information provided.'}\n\n"
    md += "### 13.4 Backup / restore / DR process\n"
    md += f"{prd.ops_devops_summary or 'Insufficient information provided.'}\n\n"
    md += "### 13.5 Required infrastructure (cloud region, nodes, DB size, etc.)\n"
    md += f"{prd.ops_devops_summary or 'Insufficient information provided.'}\n\n"

    md += "## 14. Risks, assumptions & constraints\n"
    md += f"### 14.1 Technical risks\n{_bullets(prd.risks or ['Insufficient information provided.'])}\n\n"
    md += f"### 14.2 Business risks\n{_bullets(prd.risks or ['Insufficient information provided.'])}\n\n"
    md += "### 14.3 Schedule / resource constraints\n- Insufficient information provided.\n\n"
    md += f"### 14.4 Known open questions / decisions pending\n{_bullets(prd.open_questions or ['Insufficient information provided.'])}\n\n"

    md += "## 15. References & appendices\n"
    md += "### 15.1 Related docs\n"
    md += "- Insufficient information provided.\n\n"
    md += f"### 15.2 Glossary of terms\n{_bullets(prd.glossary or ['Insufficient information provided.'])}\n\n"
    md += "### 15.3 Change history / version log\n"
    md += "| Version | Date | Author | Changes |\n"
    md += "|---------|------|--------|---------|\n"
    md += f"| 1.0 | {today} | Product Team | Initial generated PRD draft |\n"

    return md


def render_partial_prd_core(core: Stage01CoreOutput, today: str) -> str:
    md = ""
    md += "# Product Requirements Document (PRD)\n\n"

    md += "## 1. Document header / metadata\n"
    md += "### 1.1 Title and project code\nInsufficient information provided.\n\n"
    md += "### 1.2 Version, author, reviewers, approvers\n"
    md += "**Version:** 1.0\n\n"
    md += "**Author:** Product Team\n\n"
    md += "**Reviewers:** Engineering, Design, QA\n\n"
    md += "**Approvers:** Product Leadership\n\n"
    md += "### 1.3 Date of creation and last update\n"
    md += f"**Created:** {today}\n\n"
    md += f"**Last updated:** {today}\n\n"
    md += "### 1.4 Applicable teams (Product, Eng, Design, QA, DevOps, Compliance, etc.)\n"
    md += "- Product\n- Engineering\n- Design\n- QA\n- DevOps\n- Compliance\n\n"

    md += "## 2. Introduction & background\n"
    md += f"### 2.1 Problem statement / context\n{core.core_problem}\n\n"
    md += "### 2.2 Why this project is being built (pain points, business need)\n"
    md += f"{core.why_tools_fail}\n\n"
    md += "### 2.3 Link to strategy / roadmap / OKRs\nInsufficient information provided.\n\n"

    md += "## 3. Goals & success metrics\n"
    md += f"### 3.1 Product goals (what \"good\" looks like)\n{core.success_meaning}\n\n"
    md += "### 3.2 Key success metrics (KPIs)\n<!-- dv:loading -->\n\n"
    md += "### 3.3 Go-live / success criteria (what \"done\" means)\n<!-- dv:loading -->\n\n"

    md += "> Generating success metrics, personas, scope, and remaining PRD sections...\n"
    return md


def render_partial_prd_objectives(core: Stage01CoreOutput, objectives: Stage06ObjectivesOutput, today: str) -> str:
    md = render_partial_prd_core(core, today).rstrip() + "\n\n"
    md += "## 3. Goals & success metrics\n"
    md += f"### 3.1 Product goals (what \"good\" looks like)\n{objectives.primary_objective}\n\n"
    md += "### 3.2 Key success metrics (KPIs)\n"
    md += _bullets(objectives.success_metrics or ["Insufficient information provided."]) + "\n\n"
    md += "### 3.3 Go-live / success criteria (what \"done\" means)\n<!-- dv:loading -->\n\n"
    md += "> Generating personas, scope, and remaining PRD sections...\n"
    return md


def render_partial_prd_personas(core: Stage01CoreOutput, objectives: Stage06ObjectivesOutput, personas_out: Stage09PersonasOutput, today: str) -> str:
    personas = _map_personas(personas_out.personas)
    md = render_partial_prd_objectives(core, objectives, today).rstrip() + "\n\n"
    md += "## 4. Target users & personas\n"
    md += "### 4.1 Who are the main user types (end users, admins, ops, partners)\n"
    roles = [p.role.strip() for p in personas if isinstance(p.role, str) and p.role.strip()]
    md += _bullets(sorted(set(roles))) + "\n\n" if roles else "Insufficient information provided.\n\n"

    md += "### 4.2 User personas with roles, goals, and pain points\n"
    if personas:
        for p in personas:
            md += f"#### {p.name}\n"
            md += f"Role: {p.role}\n\n"
            md += f"{p.description}\n\n"
            md += f"Pain points:\n{_bullets(p.pain_points) if p.pain_points else '- Insufficient information provided.'}\n\n"
            md += f"Goals:\n{_bullets(p.goals) if p.goals else '- Insufficient information provided.'}\n\n"
    else:
        md += "Insufficient information provided.\n\n"

    md += "### 4.3 Market / customer segment overview (optional)\nInsufficient information provided.\n\n"
    md += "> Generating scope, features, NFRs, release plan, and remaining PRD sections...\n"
    return md


def _validate_markdown_structure(md: str, prd: PRDContent) -> tuple[list[str], bool]:
    errors: list[str] = []

    expected_counts = Counter(REQUIRED_SECTION_HEADINGS)
    for heading, expected_count in expected_counts.items():
        actual_count = _count_heading_lines(md, heading)
        if actual_count == 0:
            errors.append(f"MISSING:{heading}")
        elif actual_count > expected_count:
            errors.append(f"DUPLICATE:{heading}")

    # Number-marker validation is intentionally omitted here.
    # Headings are validated via REQUIRED_SECTION_HEADINGS, and section numbering can vary by renderer version.

    if re.search(r"\{\s*\"[A-Za-z0-9_]+\"\s*:", md):
        errors.append("JSON_ARTIFACT_DETECTED")

    in_scope = {x.strip().lower() for x in prd.in_scope_features if isinstance(x, str) and x.strip()}
    out_scope = {x.strip().lower() for x in prd.out_of_scope if isinstance(x, str) and x.strip()}
    overlap = in_scope & out_scope
    if overlap:
        errors.append("OUT_OF_SCOPE_LEAK_IN_SCOPE")

    return errors, len(errors) == 0


async def generate_multistep_prd(
    payload: PRDGenerateRequest,
    tenant_id: str,
    project_id: str | None = None,
    intake_id: str | None = None,
    run_id: str | None = None,
    progress_cb: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    control_cb: Callable[[], Awaitable[dict[str, bool]] | dict[str, bool] | None] | None = None,
    resume_from_stage: str | None = None,
    stage_cache: dict[str, dict[str, Any]] | None = None,
) -> PRDMultiStepResponse:
    orchestrator = PRDOrchestrator(
        tenant_id=tenant_id,
        project_id=project_id,
        intake_id=intake_id,
        run_id=run_id,
        progress_cb=progress_cb,
        control_cb=control_cb,
        resume_from_stage=resume_from_stage,
        stage_cache=stage_cache,
    )

    shared_payload = {
        "title": payload.title,
        "problem_statement": payload.problem_statement,
        "target_users": payload.target_users,
        "features": payload.features,
        "additional_notes": payload.additional_notes or "",
    }

    prd_content = await orchestrator.generate_content(shared_payload)
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    markdown = render_hierarchical_prd(
        prd_content,
        today,
        project_title=payload.title,
        project_code=project_id,
    )

    missing_sections, has_all_required_sections = _validate_markdown_structure(markdown, prd_content)
    if not has_all_required_sections:
        raise ValueError(f"Missing required sections: {missing_sections}")

    # Finalize stage for run tracking (gives the UI a definitive last step + full markdown preview).
    if progress_cb:
        try:
            await progress_cb({"stage": "stage_20_finalize", "status": "running"})
            await progress_cb(
                {
                    "stage": "stage_20_finalize",
                    "status": "completed",
                    "stage_output": {
                        "pages_estimated": 5,
                        "sections_generated": len(REQUIRED_SECTION_HEADINGS),
                        "has_all_required_sections": True,
                    },
                    "partial_markdown": markdown,
                }
            )
        except Exception:
            pass

    if project_id:
        try:
            source_id = f"{intake_id or 'unknown'}:{run_id or 'direct'}"
            source_ver = int(datetime.now(timezone.utc).timestamp())
            await store_project_source_text(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="prd",
                source_id=source_id,
                source_version=source_ver,
                text=markdown,
            )
        except Exception:
            pass

    return PRDMultiStepResponse(
        status="success",
        pages_estimated=30,
        sections_generated=REQUIRED_SECTION_HEADINGS,
        total_tokens_used=orchestrator.total_tokens_used,
        prd_markdown=markdown,
        required_sections=REQUIRED_SECTION_HEADINGS,
        missing_sections=[],
        has_all_required_sections=True,
    )
