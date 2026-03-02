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
    "# 1. Product Requirements Document (PRD)",
    "## 1.1 DecisionVault — Stage 1: Decision History Core",
    "## 2. Executive Summary",
    "## 3. Problem Statement",
    "### 3.1 The Core Problem",
    "### 3.2 Why Existing Tools Fail",
    "### 3.3 Success Will Mean",
    "## 4. Objectives and Success Metrics",
    "### 4.1 Primary Objective",
    "### 4.2 Success Metrics (Stage 1 MVP)",
    "### 4.3 Leading Indicators (Early Validation)",
    "## 5. Target Users and Personas",
    "### 5.1 Primary Persona: Sarah — Engineering Lead",
    "### 5.2 Secondary Persona: David — Product Manager",
    "### 5.3 Tertiary Persona: Maya — Startup CTO/Founder",
    "## 6. Product Scope: Stage 1 Features",
    "### 6.1 In Scope (Must Have for MVP)",
    "#### 6.1.1 Decision Capture System",
    "##### Slack Thread Capture",
    "##### Manual Document Upload",
    "##### Structured Decision Entry Form",
    "#### 6.1.2 Decision Record Storage",
    "##### Immutable Decision Records",
    "##### Versioning System",
    "##### Evidence Linking",
    "#### 6.1.3 Decision Timeline View",
    "##### Project Timeline",
    "##### Evolution Visualization",
    "#### 6.1.4 \"Why Did We...?\" Query System",
    "##### Natural Language Search",
    "##### Evidence-First Results",
    "##### Query Analytics (Admin View)",
    "#### 6.1.5 Multi-Tenant Foundation",
    "##### Tenant Architecture",
    "##### Project Hierarchy",
    "##### User Management",
    "##### Role-Based Access Control (RBAC)",
    "#### 6.1.6 License and Trial System",
    "##### Trial Mode",
    "##### Paid Licensing",
    "##### Billing Integration",
    "### 6.2 Out of Scope (Stage 1)",
    "## 7. User Stories and Use Cases",
    "### 7.1 Epic 1: Decision Capture",
    "#### US-1.1",
    "#### US-1.2",
    "#### US-1.3",
    "### 7.2 Epic 2: Decision Retrieval",
    "#### US-2.1",
    "#### US-2.2",
    "#### US-2.3",
    "### 7.3 Epic 3: Decision Management",
    "#### US-3.1",
    "#### US-3.2",
    "#### US-3.3",
    "### 7.4 Epic 4: Multi-Tenant Administration",
    "#### US-4.1",
    "#### US-4.2",
    "#### US-4.3",
    "## 8. Technical Architecture",
    "### 8.1 System Architecture Overview",
    "### 8.2 Data Model (Core Entities)",
    "#### Tenant (Organization)",
    "#### User",
    "#### Project",
    "#### Decision (Core Entity)",
    "#### Decision Relationship",
    "#### Integration (Slack)",
    "### 8.3 API Endpoints (Key Routes)",
    "#### Authentication",
    "#### Organizations",
    "#### Projects",
    "#### Decisions",
    "#### Search",
    "#### Integrations",
    "#### Billing",
    "### 8.4 Slack Integration Architecture",
    "#### OAuth Flow",
    "#### Event Subscription Flow",
    "#### Slack Bot Permissions Required",
    "#### Data Privacy",
    "### 8.5 Security and Compliance",
    "#### Data Security",
    "#### Authentication",
    "#### Access Control",
    "#### Compliance Considerations (Future)",
    "## 9. User Interface Design",
    "### 9.1 Key Screens (Wireframe Descriptions)",
    "#### Dashboard (Post-Login Landing)",
    "#### Project Timeline View",
    "#### Decision Detail View",
    "#### Search Results View",
    "#### Create/Edit Decision Form",
    "#### Admin Settings",
    "#### Billing Page",
    "### 9.2 Design Principles",
    "## 10. Dependencies and Integrations",
    "### 10.1 External Dependencies",
    "### 10.2 API Rate Limits and Handling",
    "### 10.3 Fallback Strategies",
    "## 11. Non-Functional Requirements",
    "### 11.1 Performance",
    "### 11.2 Scalability",
    "### 11.3 Reliability",
    "### 11.4 Security",
    "### 11.5 Compliance",
    "## 12. Testing Strategy",
    "### 12.1 Test Coverage Goals",
    "### 12.2 Key Test Scenarios",
    "#### Decision Capture",
    "#### Search and Retrieval",
    "#### Multi-Tenancy",
    "#### Billing",
    "#### Performance",
    "## 13. Launch Plan",
    "### 13.1 Pre-Launch (Weeks -4 to -1)",
    "### 13.2 Launch (Week 0)",
    "### 13.3 Post-Launch Monitoring",
    "## 14. Open Questions and Assumptions",
    "### 14.1 Open Questions",
    "### 14.2 Assumptions",
    "### 14.3 Risks and Mitigations",
    "## 15. Success Criteria and Definition of Done",
    "### 15.1 Stage 1 is \"Done\" When",
    "### 15.2 Stage 1 is \"Successful\" When (6-Month Post-Launch)",
    "## 16. Appendix",
    "### 16.1 Glossary",
    "### 16.2 References",
    "## 17. Document Change Log",
]

EXPECTED_NUMBER_PATHS = [
    "1.", "1.1", "2.", "3.", "3.1", "3.2", "3.3", "4.", "4.1", "4.2", "4.3", "5.", "5.1", "5.2", "5.3", "6.",
    "6.1", "6.1.1", "6.1.2", "6.1.3", "6.1.4", "6.1.5", "6.1.6", "6.2", "7.", "7.1", "7.2", "7.3", "7.4", "8.",
    "8.1", "8.2", "8.3", "8.4", "8.5", "9.", "9.1", "9.2", "10.", "10.1", "10.2", "10.3", "11.", "11.1", "11.2", "11.3", "11.4", "11.5",
    "12.", "12.1", "12.2", "13.", "13.1", "13.2", "13.3", "14.", "14.1", "14.2", "14.3", "15.", "15.1", "15.2", "16.", "16.1", "16.2", "17.",
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


class Stage1Output(BaseModel):
    executive_summary: str
    core_problem: str
    why_tools_fail: str
    success_meaning: str
    primary_objective: str
    success_metrics: list[str] = Field(default_factory=list)
    leading_indicators: list[str] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)


class Stage2Output(BaseModel):
    in_scope_features: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)


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

    async def _emit_progress(self, event: dict[str, Any]) -> None:
        if not self.progress_cb:
            return
        maybe = self.progress_cb(event)
        if hasattr(maybe, "__await__"):
            await maybe

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
    def _extract_json_block(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

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
        candidate = (text or "").strip()
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
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

    @classmethod
    def _parse_structured(cls, raw: str, schema: type[BaseModel]) -> BaseModel:
        candidate = cls._extract_json_block(raw)
        parsed: Any
        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                repaired = cls._repair_truncated_json(candidate)
                try:
                    parsed = json.loads(repaired)
                except Exception:
                    try:
                        parsed = ast.literal_eval(repaired)
                    except Exception as exc:
                        try:
                            parsed = cls._parse_loose_by_schema(repaired, schema)
                        except Exception:
                            raise ValueError(f"Unable to parse JSON output: {exc}")

        if not isinstance(parsed, dict):
            raise ValueError("Structured output must be a JSON object")
        parsed = cls._sanitize_obj(parsed)
        parsed = cls._coerce_schema_lists(parsed, schema)
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
            m = re.search(rf'"{re.escape(key)}"\s*:', text)
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

    def _bounded_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False)
        if self._estimate_tokens(raw) <= MAX_STAGE_INPUT_TOKENS:
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

        if self._estimate_tokens(json.dumps(compressed, ensure_ascii=False)) > MAX_STAGE_INPUT_TOKENS:
            raise ValueError("Input token budget exceeded after compression")
        return compressed

    async def _run_stage(
        self,
        stage_name: str,
        schema: type[BaseModel],
        stage_instruction: str,
        payload: dict[str, Any],
    ) -> StageRunResult:
        await self._check_control(stage_name)
        await self._emit_progress({"stage": stage_name, "status": "running"})

        source = self._bounded_payload(payload)
        input_json = json.dumps(source, ensure_ascii=False, indent=2)
        input_tokens = self._estimate_tokens(input_json)
        self.limiter.enforce(input_tokens=input_tokens, output_tokens=MAX_STAGE_OUTPUT_TOKENS)

        prompt = (
            f"System: {SYSTEM_CONTENT_ONLY}\n"
            f"Style: {DOC_STYLE_GUIDE}\n"
            f"Depth: {DETAIL_DEPTH_GUIDE}\n"
            f"Stage: {stage_name}\n"
            f"Task: {stage_instruction}\n"
            "Rules: output valid JSON only. no prose outside JSON. no section headers.\n"
            f"Allowed keys: {', '.join(schema.model_fields.keys())}\n\n"
            f"Input:\n{input_json}"
        )

        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(2):
            await self._check_control(stage_name)
            raw_text, used_tokens, model_name = await self._invoke_llm(prompt, MAX_STAGE_OUTPUT_TOKENS)
            self.total_tokens_used += used_tokens
            try:
                parsed = self._parse_structured(raw_text, schema)
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
                    }
                )
                return StageRunResult(output=parsed, input_tokens=input_tokens, output_tokens=output_tokens, retry_count=retry_count)
            except Exception as exc:
                last_error = exc
                retry_count = 1
                logger.warning("prd_stage_validation_failed stage=%s attempt=%s error=%s", stage_name, attempt + 1, exc)

        await self._emit_progress({"stage": stage_name, "status": "failed", "error": str(last_error)})
        raise ValueError(f"{stage_name} failed: {last_error}")

    async def generate_content(self, payload: dict[str, Any]) -> PRDContent:
        if self.project_id:
            try:
                await sync_project_knowledge_chunks(tenant_id=self.tenant_id, project_id=self.project_id)
                retrieval_query = "\n".join(
                    [
                        f"title: {payload.get('title')}",
                        f"problem_statement: {payload.get('problem_statement')}",
                        "target_users: " + ", ".join(str(x) for x in (payload.get("target_users") or [])),
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

        stage1 = await self._run_stage(
            stage_name="stage_1_core_context",
            schema=Stage1Output,
            stage_instruction=(
                "Fill only these fields: executive_summary, core_problem, why_tools_fail, success_meaning, "
                "primary_objective, success_metrics, leading_indicators, personas. "
                "For personas, create exactly 3 entries aligned to engineering lead, product manager, CTO/founder. "
                "Executive summary and problem fields must be detailed and decision-history focused."
            ),
            payload={
                "title": payload["title"],
                "problem_statement": payload["problem_statement"],
                "target_users": payload["target_users"],
                "features": payload["features"],
                "additional_notes": payload.get("additional_notes", ""),
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
        )

        stage2 = await self._run_stage(
            stage_name="stage_2_scope_user_stories",
            schema=Stage2Output,
            stage_instruction=(
                "Fill only in_scope_features, out_of_scope, user_stories. "
                "Return exactly 12 user stories with IDs US-1.1..US-4.3 and practical acceptance criteria. "
                "Each acceptance criteria list should include measurable, testable statements."
            ),
            payload={
                "title": payload["title"],
                "problem_statement": payload["problem_statement"],
                "target_users": payload["target_users"],
                "features": payload["features"],
                "additional_notes": payload.get("additional_notes", ""),
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
        )

        stage3 = await self._run_stage(
            stage_name="stage_3_architecture",
            schema=Stage3Output,
            stage_instruction=(
                "Fill architecture_summary, data_model_summary, api_summary, slack_integration_summary, security_summary. "
                "Keep statements implementation-oriented and consistent with provided stack and features only. "
                "Include concrete architecture boundaries, key entities, and API route patterns."
            ),
            payload={
                "title": payload["title"],
                "problem_statement": payload["problem_statement"],
                "target_users": payload["target_users"],
                "features": payload["features"],
                "additional_notes": payload.get("additional_notes", ""),
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
        )

        stage4 = await self._run_stage(
            stage_name="stage_4_delivery_quality",
            schema=Stage4Output,
            stage_instruction=(
                "Fill ui_summary, dependencies_summary, non_functional_summary, testing_summary, launch_plan_summary, "
                "open_questions, assumptions, risks, definition_of_done, glossary. "
                "Provide actionable release-quality detail, not short placeholders."
            ),
            payload={
                "title": payload["title"],
                "problem_statement": payload["problem_statement"],
                "target_users": payload["target_users"],
                "features": payload["features"],
                "additional_notes": payload.get("additional_notes", ""),
                "retrieved_project_knowledge_chunks": self.retrieved_chunks,
            },
        )

        return PRDContent(
            executive_summary=stage1.output.executive_summary,
            core_problem=stage1.output.core_problem,
            why_tools_fail=stage1.output.why_tools_fail,
            success_meaning=stage1.output.success_meaning,
            primary_objective=stage1.output.primary_objective,
            success_metrics=stage1.output.success_metrics,
            leading_indicators=stage1.output.leading_indicators,
            personas=stage1.output.personas,
            in_scope_features=stage2.output.in_scope_features,
            out_of_scope=stage2.output.out_of_scope,
            user_stories=stage2.output.user_stories,
            architecture_summary=stage3.output.architecture_summary,
            data_model_summary=stage3.output.data_model_summary,
            api_summary=stage3.output.api_summary,
            slack_integration_summary=stage3.output.slack_integration_summary,
            security_summary=stage3.output.security_summary,
            ui_summary=stage4.output.ui_summary,
            dependencies_summary=stage4.output.dependencies_summary,
            non_functional_summary=stage4.output.non_functional_summary,
            testing_summary=stage4.output.testing_summary,
            launch_plan_summary=stage4.output.launch_plan_summary,
            open_questions=stage4.output.open_questions,
            assumptions=stage4.output.assumptions,
            risks=stage4.output.risks,
            definition_of_done=stage4.output.definition_of_done,
            glossary=stage4.output.glossary,
        )


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


def render_hierarchical_prd(prd: PRDContent, today: str) -> str:
    personas = _map_personas(prd.personas)
    stories = _story_map(prd.user_stories)

    md = ""
    md += "# 1. Product Requirements Document (PRD)\n"
    md += "## 1.1 DecisionVault — Stage 1: Decision History Core\n"
    md += "**Version:** 1.0\n"
    md += f"**Date:** {today}\n"
    md += "**Status:** Draft for Development\n"
    md += "**Owner:** Product Team\n"
    md += "**Contributors:** Engineering, Design\n\n"

    md += "## 2. Executive Summary\n"
    md += f"{prd.executive_summary}\n\n"

    md += "## 3. Problem Statement\n"
    md += f"### 3.1 The Core Problem\n{prd.core_problem}\n\n"
    md += f"### 3.2 Why Existing Tools Fail\n{prd.why_tools_fail}\n\n"
    md += f"### 3.3 Success Will Mean\n{prd.success_meaning}\n\n"

    md += "## 4. Objectives and Success Metrics\n"
    md += f"### 4.1 Primary Objective\n{prd.primary_objective}\n\n"
    md += "### 4.2 Success Metrics (Stage 1 MVP)\n"
    md += _bullets(prd.success_metrics) + "\n\n"
    md += "### 4.3 Leading Indicators (Early Validation)\n"
    md += _bullets(prd.leading_indicators) + "\n\n"

    md += "## 5. Target Users and Personas\n"
    md += "### 5.1 Primary Persona: Sarah — Engineering Lead\n"
    md += f"{personas[0].description}\nPain Points:\n{_bullets(personas[0].pain_points)}\nGoals:\n{_bullets(personas[0].goals)}\n\n"
    md += "### 5.2 Secondary Persona: David — Product Manager\n"
    md += f"{personas[1].description}\nPain Points:\n{_bullets(personas[1].pain_points)}\nGoals:\n{_bullets(personas[1].goals)}\n\n"
    md += "### 5.3 Tertiary Persona: Maya — Startup CTO/Founder\n"
    md += f"{personas[2].description}\nPain Points:\n{_bullets(personas[2].pain_points)}\nGoals:\n{_bullets(personas[2].goals)}\n\n"

    md += "## 6. Product Scope: Stage 1 Features\n"
    md += "### 6.1 In Scope (Must Have for MVP)\n"
    md += "#### 6.1.1 Decision Capture System\n"
    md += f"##### Slack Thread Capture\n{_bullets(_feature_slice(prd.in_scope_features, 0, 3))}\n"
    md += f"##### Manual Document Upload\n{_bullets(_feature_slice(prd.in_scope_features, 3, 3))}\n"
    md += f"##### Structured Decision Entry Form\n{_bullets(_feature_slice(prd.in_scope_features, 6, 3))}\n"

    md += "#### 6.1.2 Decision Record Storage\n"
    md += f"##### Immutable Decision Records\n{_bullets(_feature_slice(prd.in_scope_features, 9, 2))}\n"
    md += f"##### Versioning System\n{_bullets(_feature_slice(prd.in_scope_features, 11, 2))}\n"
    md += f"##### Evidence Linking\n{_bullets(_feature_slice(prd.in_scope_features, 13, 2))}\n"

    md += "#### 6.1.3 Decision Timeline View\n"
    md += f"##### Project Timeline\n{_bullets(_feature_slice(prd.in_scope_features, 15, 2))}\n"
    md += f"##### Evolution Visualization\n{_bullets(_feature_slice(prd.in_scope_features, 17, 2))}\n"

    md += "#### 6.1.4 \"Why Did We...?\" Query System\n"
    md += f"##### Natural Language Search\n{_bullets(_feature_slice(prd.in_scope_features, 19, 2))}\n"
    md += f"##### Evidence-First Results\n{_bullets(_feature_slice(prd.in_scope_features, 21, 2))}\n"
    md += f"##### Query Analytics (Admin View)\n{_bullets(_feature_slice(prd.in_scope_features, 23, 2))}\n"

    md += "#### 6.1.5 Multi-Tenant Foundation\n"
    md += f"##### Tenant Architecture\n{_bullets(_split_sentences(prd.architecture_summary))}\n"
    md += f"##### Project Hierarchy\n{_bullets(_split_sentences(prd.data_model_summary))}\n"
    md += f"##### User Management\n{_bullets(_split_sentences(prd.api_summary))}\n"
    md += f"##### Role-Based Access Control (RBAC)\n{_bullets(_split_sentences(prd.security_summary))}\n"

    md += "#### 6.1.6 License and Trial System\n"
    md += f"##### Trial Mode\n{_bullets(_feature_slice(prd.in_scope_features, 25, 2))}\n"
    md += f"##### Paid Licensing\n{_bullets(_feature_slice(prd.in_scope_features, 27, 2))}\n"
    md += f"##### Billing Integration\n{_bullets(_split_sentences(prd.dependencies_summary))}\n"

    md += "### 6.2 Out of Scope (Stage 1)\n"
    md += _bullets(prd.out_of_scope) + "\n\n"

    md += "## 7. User Stories and Use Cases\n"
    md += "### 7.1 Epic 1: Decision Capture\n"
    for sid in ["US-1.1", "US-1.2", "US-1.3"]:
        us = stories[sid]
        md += f"#### {sid}\n{us.description}\nAcceptance Criteria:\n{_bullets(us.acceptance_criteria)}\n"

    md += "### 7.2 Epic 2: Decision Retrieval\n"
    for sid in ["US-2.1", "US-2.2", "US-2.3"]:
        us = stories[sid]
        md += f"#### {sid}\n{us.description}\nAcceptance Criteria:\n{_bullets(us.acceptance_criteria)}\n"

    md += "### 7.3 Epic 3: Decision Management\n"
    for sid in ["US-3.1", "US-3.2", "US-3.3"]:
        us = stories[sid]
        md += f"#### {sid}\n{us.description}\nAcceptance Criteria:\n{_bullets(us.acceptance_criteria)}\n"

    md += "### 7.4 Epic 4: Multi-Tenant Administration\n"
    for sid in ["US-4.1", "US-4.2", "US-4.3"]:
        us = stories[sid]
        md += f"#### {sid}\n{us.description}\nAcceptance Criteria:\n{_bullets(us.acceptance_criteria)}\n"

    md += "## 8. Technical Architecture\n"
    md += f"### 8.1 System Architecture Overview\n{prd.architecture_summary}\n\n"

    md += "### 8.2 Data Model (Core Entities)\n"
    md += f"#### Tenant (Organization)\n{prd.data_model_summary}\n"
    md += f"#### User\n{prd.data_model_summary}\n"
    md += f"#### Project\n{prd.data_model_summary}\n"
    md += f"#### Decision (Core Entity)\n{prd.data_model_summary}\n"
    md += f"#### Decision Relationship\n{prd.data_model_summary}\n"
    md += f"#### Integration (Slack)\n{prd.data_model_summary}\n"

    md += "### 8.3 API Endpoints (Key Routes)\n"
    for group in ["Authentication", "Organizations", "Projects", "Decisions", "Search", "Integrations", "Billing"]:
        md += f"#### {group}\n{prd.api_summary}\n"

    md += "### 8.4 Slack Integration Architecture\n"
    md += f"#### OAuth Flow\n{prd.slack_integration_summary}\n"
    md += f"#### Event Subscription Flow\n{prd.slack_integration_summary}\n"
    md += f"#### Slack Bot Permissions Required\n{prd.slack_integration_summary}\n"
    md += f"#### Data Privacy\n{prd.slack_integration_summary}\n"

    md += "### 8.5 Security and Compliance\n"
    md += f"#### Data Security\n{prd.security_summary}\n"
    md += f"#### Authentication\n{prd.security_summary}\n"
    md += f"#### Access Control\n{prd.security_summary}\n"
    md += f"#### Compliance Considerations (Future)\n{prd.security_summary}\n\n"

    md += "## 9. User Interface Design\n"
    md += "### 9.1 Key Screens (Wireframe Descriptions)\n"
    for screen in [
        "Dashboard (Post-Login Landing)",
        "Project Timeline View",
        "Decision Detail View",
        "Search Results View",
        "Create/Edit Decision Form",
        "Admin Settings",
        "Billing Page",
    ]:
        md += f"#### {screen}\n{prd.ui_summary}\n"
    md += f"### 9.2 Design Principles\n{prd.ui_summary}\n\n"

    md += "## 10. Dependencies and Integrations\n"
    md += f"### 10.1 External Dependencies\n{prd.dependencies_summary}\n"
    md += f"### 10.2 API Rate Limits and Handling\n{prd.dependencies_summary}\n"
    md += f"### 10.3 Fallback Strategies\n{prd.dependencies_summary}\n\n"

    md += "## 11. Non-Functional Requirements\n"
    md += f"### 11.1 Performance\n{prd.non_functional_summary}\n"
    md += f"### 11.2 Scalability\n{prd.non_functional_summary}\n"
    md += f"### 11.3 Reliability\n{prd.non_functional_summary}\n"
    md += f"### 11.4 Security\n{prd.non_functional_summary}\n"
    md += f"### 11.5 Compliance\n{prd.non_functional_summary}\n\n"

    md += "## 12. Testing Strategy\n"
    md += f"### 12.1 Test Coverage Goals\n{prd.testing_summary}\n"
    md += "### 12.2 Key Test Scenarios\n"
    for scenario in ["Decision Capture", "Search and Retrieval", "Multi-Tenancy", "Billing", "Performance"]:
        md += f"#### {scenario}\n{prd.testing_summary}\n"

    md += "## 13. Launch Plan\n"
    md += f"### 13.1 Pre-Launch (Weeks -4 to -1)\n{prd.launch_plan_summary}\n"
    md += f"### 13.2 Launch (Week 0)\n{prd.launch_plan_summary}\n"
    md += f"### 13.3 Post-Launch Monitoring\n{prd.launch_plan_summary}\n\n"

    md += "## 14. Open Questions and Assumptions\n"
    md += f"### 14.1 Open Questions\n{_bullets(prd.open_questions)}\n"
    md += f"### 14.2 Assumptions\n{_bullets(prd.assumptions)}\n"
    md += f"### 14.3 Risks and Mitigations\n{_bullets(prd.risks)}\n\n"

    md += "## 15. Success Criteria and Definition of Done\n"
    md += f"### 15.1 Stage 1 is \"Done\" When\n{_bullets(prd.definition_of_done)}\n"
    md += f"### 15.2 Stage 1 is \"Successful\" When (6-Month Post-Launch)\n{_bullets(prd.leading_indicators)}\n\n"

    md += "## 16. Appendix\n"
    md += f"### 16.1 Glossary\n{_bullets(prd.glossary)}\n"
    md += "### 16.2 References\n"
    md += "- [1] User-provided project context from requirement intake.\n"
    md += "- [2] Product constraints and notes supplied in the generation payload.\n\n"

    md += "## 17. Document Change Log\n"
    md += "| Version | Date | Author | Changes |\n"
    md += "|---------|------|--------|---------|\n"
    md += f"| 1.0 | {today} | Product Team | Initial generated PRD draft |\n"

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

    for marker in EXPECTED_NUMBER_PATHS:
        if not re.search(rf"\b{re.escape(marker)}", md):
            errors.append(f"NUMBER_MISSING:{marker}")

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
) -> PRDMultiStepResponse:
    orchestrator = PRDOrchestrator(
        tenant_id=tenant_id,
        project_id=project_id,
        intake_id=intake_id,
        run_id=run_id,
        progress_cb=progress_cb,
        control_cb=control_cb,
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
    markdown = render_hierarchical_prd(prd_content, today)

    missing_sections, has_all_required_sections = _validate_markdown_structure(markdown, prd_content)
    if not has_all_required_sections:
        raise ValueError(f"Missing required sections: {missing_sections}")

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
        pages_estimated=5,
        sections_generated=REQUIRED_SECTION_HEADINGS,
        total_tokens_used=orchestrator.total_tokens_used,
        prd_markdown=markdown,
        required_sections=REQUIRED_SECTION_HEADINGS,
        missing_sections=[],
        has_all_required_sections=True,
    )
