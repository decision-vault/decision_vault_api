from __future__ import annotations

import json
import logging
import re
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Type

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings
from app.schemas.prd_generation import PRDGenerateRequest, PRDMultiStepResponse
from app.services.llm_usage_service import log_llm_usage
from app.services.token_limiter import TokenBudget, TokenLimiter


logger = logging.getLogger("decisionvault.prd.multistep")

STRICT_PROMPT = (
    "Only use provided information. Do not invent statistics, integrations, or technologies. "
    "If data is missing, write 'Insufficient information provided'. "
    "Write concrete, professional content and avoid placeholder wording. "
    "Never output JSON fragments, key names, braces, or escaped markdown."
)

DETAIL_PROMPT = (
    "Expand depth and clarity while preserving meaning. "
    "Do not introduce new facts not present in input. "
    "Prefer implementation-ready wording."
)

DOC_STYLE_GUIDE = (
    "Output must be professional PRD-quality Markdown-ready prose, never JSON-like prose. "
    "Each section value must be clean plain text suitable for direct insertion into Markdown. "
    "No braces, no quoted key/value output, no code fences, no escaped newlines (\\\\n), and no LaTeX commands. "
    "Use concise subheadings, dense but readable bullets, and measurable language when present in input. "
    "Use plain Markdown tables only when requested."
)

KNOWN_INTEGRATIONS = {
    "slack",
    "jira",
    "trello",
    "salesforce",
    "stripe",
    "razorpay",
    "github",
    "notion",
    "google drive",
    "figma",
}

MAX_STAGE_INPUT_TOKENS = 3000
MAX_STAGE_OUTPUT_TOKENS = 3000
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


class TextSection(BaseModel):
    content: str


class ListSection(BaseModel):
    items: list[str]


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
        progress_cb: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ):
        self.tenant_id = tenant_id
        self.progress_cb = progress_cb
        self.total_tokens_used = 0
        self.sections_generated: list[str] = []
        self.limiter = TokenLimiter(
            TokenBudget(
                max_input_tokens=MAX_STAGE_INPUT_TOKENS,
                max_output_tokens=MAX_STAGE_OUTPUT_TOKENS,
            )
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\\s*", "", cleaned)
            cleaned = re.sub(r"\\s*```$", "", cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_json_block(cls, text: str) -> str:
        cleaned = cls._strip_code_fence(text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return cleaned
        return cleaned[start : end + 1]

    @staticmethod
    def _sanitize_generated_text(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return "Insufficient information provided"
        s = s.replace("\\n", "\n").replace("\\t", " ")
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = re.sub(r'^\{\s*"?(content|text|items)"?\s*:\s*', "", s, flags=re.IGNORECASE)
        s = re.sub(r"\}\s*$", "", s)
        s = s.strip().strip('"').strip("'")
        s = re.sub(r"\s+", " ", s).strip()
        return s or "Insufficient information provided"

    async def _emit_progress(self, event: dict[str, Any]) -> None:
        if not self.progress_cb:
            return
        maybe = self.progress_cb(event)
        if hasattr(maybe, "__await__"):
            await maybe

    @classmethod
    def _fallback_parse(cls, schema: Type[BaseModel], raw_text: str) -> BaseModel:
        cleaned = cls._strip_code_fence(raw_text)
        if schema is TextSection:
            return TextSection(content=cls._sanitize_generated_text(cleaned))
        if schema is ListSection:
            lines = [line.strip(" -•\t") for line in cleaned.splitlines()]
            items = [line for line in lines if line]
            if not items:
                items = [part.strip() for part in re.split(r"[;\n]", cleaned) if part.strip()]
            cleaned_items = [cls._sanitize_generated_text(i) for i in items]
            return ListSection(items=cleaned_items or ["Insufficient information provided"])
        return schema.model_validate({})

    @classmethod
    def _parse_stage_output(cls, schema: Type[BaseModel], raw_text: str) -> BaseModel:
        candidate = cls._extract_json_block(raw_text)
        try:
            parsed = json.loads(candidate)
            return cls._coerce_stage_output(schema, parsed)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                return cls._coerce_stage_output(schema, parsed)
            except Exception:
                return cls._fallback_parse(schema, raw_text)

    @staticmethod
    def _collect_strings(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            clean = value.strip()
            return [clean] if clean else []
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                out.extend(PRDOrchestrator._collect_strings(item))
            return out
        if isinstance(value, dict):
            out: list[str] = []
            for key in ("content", "text", "summary", "description", "title", "value"):
                if key in value:
                    out.extend(PRDOrchestrator._collect_strings(value.get(key)))
            for k, v in value.items():
                if k in {"content", "text", "summary", "description", "title", "value"}:
                    continue
                out.extend(PRDOrchestrator._collect_strings(v))
            return out
        return []

    @classmethod
    def _coerce_stage_output(cls, schema: Type[BaseModel], parsed_json: Any) -> BaseModel:
        if schema is TextSection:
            if isinstance(parsed_json, dict) and isinstance(parsed_json.get("content"), str):
                return TextSection(content=cls._sanitize_generated_text(parsed_json["content"]))
            strings = cls._collect_strings(parsed_json)
            return TextSection(
                content=cls._sanitize_generated_text(strings[0] if strings else "Insufficient information provided")
            )

        if schema is ListSection:
            if isinstance(parsed_json, dict) and isinstance(parsed_json.get("items"), list):
                items = [cls._sanitize_generated_text(str(item)) for item in parsed_json["items"] if str(item).strip()]
                return ListSection(items=items or ["Insufficient information provided"])
            items = cls._collect_strings(parsed_json)
            deduped: list[str] = []
            seen: set[str] = set()
            for item in items:
                cleaned_item = cls._sanitize_generated_text(item)
                if cleaned_item not in seen:
                    seen.add(cleaned_item)
                    deduped.append(cleaned_item)
            return ListSection(items=deduped or ["Insufficient information provided"])

        return schema.model_validate(parsed_json)

    @staticmethod
    def _compress_text(text: str, max_words: int) -> str:
        words = (text or "").split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def _compress_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        compressed = dict(payload)
        for key, value in list(compressed.items()):
            if isinstance(value, str):
                compressed[key] = self._compress_text(value, 250)
            elif isinstance(value, list):
                compact_items: list[str] = []
                for item in value:
                    if not isinstance(item, str):
                        continue
                    compact_items.append(self._compress_text(item, 30))
                    if len(compact_items) >= 10:
                        break
                compressed[key] = compact_items
        return compressed

    @staticmethod
    def _detect_hallucination(input_text: str, output_text: str) -> bool:
        src = input_text.lower()
        out = output_text.lower()

        in_numbers = set(re.findall(r"\\d+(?:[\\.,]\\d+)?", src))
        out_numbers = set(re.findall(r"\\d+(?:[\\.,]\\d+)?", out))
        if out_numbers - in_numbers:
            return True

        in_integrations = {name for name in KNOWN_INTEGRATIONS if name in src}
        out_integrations = {name for name in KNOWN_INTEGRATIONS if name in out}
        if out_integrations - in_integrations:
            return True

        if "%" in out and "%" not in src:
            return True
        if re.search(r"[$€£₹]", out) and not re.search(r"[$€£₹]", src):
            return True
        return False

    def _provider_config(self, model_name: str) -> tuple[str, str, str | None, str]:
        provider = (settings.llm_provider or "").strip().lower()
        if provider == "lmstudio":
            return (
                settings.lmstudio_model or model_name,
                settings.llm_api_key or "lm-studio",
                settings.lmstudio_base_url,
                "lmstudio",
            )
        if provider == "huggingface":
            return (
                settings.hf_openai_model or model_name,
                settings.hf_api_token,
                settings.hf_router_base_url,
                "huggingface",
            )
        return (model_name, settings.llm_api_key, settings.llm_base_url, "default")

    async def _invoke_llm(self, prompt: str, output_tokens: int) -> tuple[str, int, str]:
        model_name, api_key, base_url, provider = self._provider_config(settings.llm_model)
        if not api_key:
            raise ValueError("LLM API key not configured")

        if provider == "lmstudio":
            url = f"{(base_url or '').rstrip('/')}{settings.lmstudio_chat_path}"
            body = {
                "model": model_name,
                "input": prompt,
                "temperature": 0.2,
                "context_length": max(6000, MAX_STAGE_INPUT_TOKENS + output_tokens),
            }
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                payload = response.json()
            output = payload.get("output")
            text = ""
            if isinstance(output, list):
                lines = []
                for item in output:
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        lines.append(item["content"].strip())
                text = "\n".join([line for line in lines if line]).strip()
            usage = payload.get("stats") if isinstance(payload, dict) else {}
            if isinstance(usage, dict):
                tokens_used = int((usage.get("input_tokens") or 0) + (usage.get("total_output_tokens") or 0))
            else:
                tokens_used = self._estimate_tokens(prompt + "\n" + text)
            return text, tokens_used, model_name

        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            top_p=0.8,
            max_tokens=output_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        message = await llm.ainvoke(prompt)
        content = (getattr(message, "content", "") or "").strip()
        usage = getattr(message, "response_metadata", {}) or {}
        token_usage = usage.get("token_usage") or usage.get("usage") or {}
        tokens_used = int(token_usage.get("total_tokens") or 0) or self._estimate_tokens(prompt + "\n" + content)
        return content, tokens_used, model_name

    async def _run_stage(
        self,
        stage_name: str,
        schema: Type[BaseModel],
        payload: dict[str, Any],
        stage_instructions: str,
    ) -> StageRunResult:
        logger.warning("prd_multistep_stage_start stage=%s", stage_name)
        await self._emit_progress(
            {
                "stage": stage_name,
                "status": "running",
            }
        )
        source_payload = payload
        input_json = json.dumps(source_payload, ensure_ascii=True)
        input_tokens = self._estimate_tokens(input_json)
        if input_tokens > MAX_STAGE_INPUT_TOKENS:
            source_payload = self._compress_input(source_payload)
            input_json = json.dumps(source_payload, ensure_ascii=True)
            input_tokens = self._estimate_tokens(input_json)
        self.limiter.enforce(input_tokens=input_tokens, output_tokens=MAX_STAGE_OUTPUT_TOKENS)

        base_prompt = (
            f"System Rules: {STRICT_PROMPT}\\n"
            f"Document Standard: {DOC_STYLE_GUIDE}\\n"
            f"Stage: {stage_name}\\n"
            f"Instructions: {stage_instructions}\\n"
            f"Return JSON object only with this schema keys: {', '.join(schema.model_fields.keys())}.\\n"
            "Do not include markdown, explanations, code fences, or extra keys.\\n\\n"
            f"Input JSON:\\n{json.dumps(source_payload, ensure_ascii=False, indent=2)}"
        )

        retry_count = 0
        last_error: Exception | None = None
        for attempt in range(2):
            stronger = "" if attempt == 0 else "\\nIMPORTANT: You introduced unsupported details previously. Remove all unsupported details."
            prompt = base_prompt + stronger
            raw_text, used_tokens, model_name = await self._invoke_llm(prompt, MAX_STAGE_OUTPUT_TOKENS)
            self.total_tokens_used += used_tokens
            try:
                parsed = self._parse_stage_output(schema, raw_text)
                if self._detect_hallucination(input_json, parsed.model_dump_json()):
                    if attempt == 0:
                        retry_count = 1
                        continue
                    raise ValueError("Hallucination suspected")
                await log_llm_usage(
                    feature=f"prd_multistep:{stage_name}",
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=max(1, used_tokens - input_tokens),
                    tenant_id=self.tenant_id,
                    stage_name=stage_name,
                    retry_count=retry_count,
                )
                logger.warning(
                    "prd_multistep_stage_done stage=%s input_tokens=%s output_tokens=%s retry=%s",
                    stage_name,
                    input_tokens,
                    used_tokens,
                    retry_count,
                )
                await self._emit_progress(
                    {
                        "stage": stage_name,
                        "status": "completed",
                        "input_tokens": input_tokens,
                        "output_tokens": max(1, used_tokens - input_tokens),
                        "retry_count": retry_count,
                    }
                )
                return StageRunResult(parsed, input_tokens, used_tokens, retry_count)
            except Exception as exc:
                last_error = exc
                retry_count = 1

        await self._emit_progress(
            {
                "stage": stage_name,
                "status": "failed",
                "error": str(last_error),
            }
        )
        raise ValueError(f"{stage_name} failed: {last_error}")

    async def _run_text_section(self, stage_name: str, payload: dict[str, Any], instruction: str) -> str:
        result = await self._run_stage(stage_name, TextSection, payload, instruction)
        content = re.sub(r"\s+", " ", (result.output.content or "")).strip()
        return content if content else "Insufficient information provided"

    async def _run_list_section(self, stage_name: str, payload: dict[str, Any], instruction: str) -> list[str]:
        result = await self._run_stage(stage_name, ListSection, payload, instruction)
        items = [item.strip() for item in (result.output.items or []) if isinstance(item, str) and item.strip()]
        return self._clean_list_items(items)

    async def generate_core_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "executive_summary": await self._run_text_section(
                "stage_1_executive_summary",
                {"title": payload["title"], "problem_statement": payload["problem_statement"], "target_users": payload["target_users"]},
                "Write Executive Summary in 2-3 paragraphs. Include: product purpose, value proposition, and stage focus.",
            ),
            "problem_statement": await self._run_text_section(
                "stage_2_problem_statement",
                {"problem_statement": payload["problem_statement"], "additional_notes": payload.get("additional_notes", "")},
                "Write Problem Statement with two subparts: 'The Core Problem' and 'Why Existing Approaches Fail'.",
            ),
            "target_users_personas": await self._run_list_section(
                "stage_3_target_users_personas",
                {"target_users": payload["target_users"], "problem_statement": payload["problem_statement"]},
                "List 3-6 personas. Each item should include role + pain + goal in one line.",
            ),
            "objectives_success_metrics": await self._run_list_section(
                "stage_4_objectives_success_metrics",
                {"features": payload["features"], "additional_notes": payload.get("additional_notes", "")},
                "List 5-10 objectives and success metrics. Prefer measurable targets if present in input.",
            ),
        }
        if self._should_run_detail_pass(payload):
            sections["executive_summary"] = await self._expand_text_section(
                "stage_1b_executive_summary_expand",
                payload,
                sections["executive_summary"],
                "Improve precision and business clarity in the executive summary.",
            )
            sections["problem_statement"] = await self._expand_text_section(
                "stage_2b_problem_statement_expand",
                payload,
                sections["problem_statement"],
                "Refine pain points, current-state gaps, and impact without adding new facts.",
            )
            sections["objectives_success_metrics"] = await self._expand_list_section(
                "stage_4b_objectives_success_metrics_expand",
                payload,
                sections["objectives_success_metrics"],
                "Improve objective/metric quality; keep measurable wording where possible.",
            )
        return sections

    async def generate_feature_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "feature_overview": await self._run_list_section(
                "stage_5_feature_overview",
                {"features": payload["features"], "problem_statement": payload["problem_statement"]},
                "List 8-12 Feature Overview items, grouped logically by capability in wording.",
            ),
            "out_of_scope": await self._run_list_section(
                "stage_6_out_of_scope",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List Out-of-scope items only if explicitly mentioned or safely implied by provided context.",
            ),
        }
        if self._should_run_detail_pass(payload):
            sections["feature_overview"] = await self._expand_list_section(
                "stage_5b_feature_overview_expand",
                payload,
                sections["feature_overview"],
                "Increase functional clarity and acceptance-oriented wording for each feature.",
            )
        return sections

    async def generate_architecture_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "architecture_decisions": await self._run_list_section(
                "stage_7_architecture_decisions",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List 6-12 Architecture Decisions using decision-style phrasing (what + why).",
            ),
            "technical_architecture": await self._run_list_section(
                "stage_8_technical_architecture",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List 8-12 Technical Architecture details: components, data flow, and integration boundaries.",
            ),
            "deployment_strategy": await self._run_list_section(
                "stage_9_deployment_strategy",
                {"additional_notes": payload.get("additional_notes", "")},
                "List 5-10 Deployment Strategy details: environment, release flow, rollback, observability.",
            ),
        }
        if self._should_run_detail_pass(payload):
            sections["architecture_decisions"] = await self._expand_list_section(
                "stage_7b_architecture_decisions_expand",
                payload,
                sections["architecture_decisions"],
                "Improve decision rationale and operational specificity.",
            )
            sections["technical_architecture"] = await self._expand_list_section(
                "stage_8b_technical_architecture_expand",
                payload,
                sections["technical_architecture"],
                "Improve module boundaries, interfaces, and data flow clarity.",
            )
        return sections

    async def generate_nonfunctional_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "non_functional_requirements": await self._run_list_section(
                "stage_10_non_functional_requirements",
                {"additional_notes": payload.get("additional_notes", ""), "problem_statement": payload["problem_statement"]},
                "List 8-12 Non-Functional Requirements with concrete quality expectations.",
            ),
            "security_compliance": await self._run_list_section(
                "stage_11_security_compliance",
                {"additional_notes": payload.get("additional_notes", "")},
                "List 6-10 Security & Compliance requirements from provided context only.",
            ),
        }
        if self._should_run_detail_pass(payload):
            sections["non_functional_requirements"] = await self._expand_list_section(
                "stage_10b_non_functional_requirements_expand",
                payload,
                sections["non_functional_requirements"],
                "Improve measurable quality criteria and operational validation language.",
            )
        return sections

    async def generate_risk_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "scalability_considerations": await self._run_list_section(
                "stage_12_scalability_considerations",
                {"features": payload["features"], "additional_notes": payload.get("additional_notes", "")},
                "List 6-10 Scalability Considerations aligned with expected usage patterns.",
            ),
            "risks_mitigation": await self._run_list_section(
                "stage_13_risks_mitigation",
                {"features": payload["features"], "additional_notes": payload.get("additional_notes", "")},
                "List 6-12 Risks & Mitigation items as actionable paired bullets.",
            ),
            "constraints": await self._run_list_section(
                "stage_14_constraints",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List 6-10 delivery/business/technical constraints.",
            ),
            "definition_of_done": await self._run_list_section(
                "stage_15_definition_of_done",
                {"features": payload["features"], "problem_statement": payload["problem_statement"]},
                "List 8-12 Definition of Done criteria that are verifiable and release-ready.",
            ),
        }
        if self._should_run_detail_pass(payload):
            sections["risks_mitigation"] = await self._expand_list_section(
                "stage_13b_risks_mitigation_expand",
                payload,
                sections["risks_mitigation"],
                "Improve risk articulation and mitigation actionability.",
            )
            sections["definition_of_done"] = await self._expand_list_section(
                "stage_15b_definition_of_done_expand",
                payload,
                sections["definition_of_done"],
                "Improve verifiability and release-readiness of done criteria.",
            )
        return sections

    @staticmethod
    def _bullets(items: list[str]) -> str:
        clean = [item.strip() for item in items if item and item.strip()]
        return "\\n".join([f"- {item}" for item in clean]) if clean else "- Insufficient information provided"

    @staticmethod
    def _phased_plan(
        feature_items: list[str],
        architecture_items: list[str],
        nonfunctional_items: list[str],
    ) -> str:
        f = [x for x in feature_items if x and x.strip()]
        a = [x for x in architecture_items if x and x.strip()]
        n = [x for x in nonfunctional_items if x and x.strip()]
        p1 = f[:3] + a[:2]
        p2 = f[3:7] + a[2:5]
        p3 = f[7:10] + n[:3]
        return (
            "### Phase 1 — Foundation\n"
            + ("\n".join([f"- {x}" for x in (p1 or ["Insufficient information provided"])]) + "\n\n")
            + "### Phase 2 — Core Workflow Completion\n"
            + ("\n".join([f"- {x}" for x in (p2 or ["Insufficient information provided"])]) + "\n\n")
            + "### Phase 3 — Hardening and Launch Readiness\n"
            + ("\n".join([f"- {x}" for x in (p3 or ["Insufficient information provided"])]))
        )

    @staticmethod
    def _metrics_table(metrics: list[str]) -> str:
        rows = [m for m in metrics if m and m.strip()]
        if not rows:
            return "| Metric | Target |\n|---|---|\n| Insufficient information provided | Insufficient information provided |"
        lines = ["| Metric | Target |", "|---|---|"]
        for item in rows[:10]:
            parts = [p.strip() for p in re.split(r"[:\-]", item, maxsplit=1) if p.strip()]
            if len(parts) == 2:
                metric, target = parts
            else:
                metric, target = item.strip(), "As provided"
            lines.append(f"| {metric} | {target} |")
        return "\n".join(lines)

    @staticmethod
    def _risk_table(risks: list[str]) -> str:
        rows = [r for r in risks if r and r.strip()]
        if not rows:
            return "| Risk | Mitigation |\n|---|---|\n| Insufficient information provided | Insufficient information provided |"
        lines = ["| Risk | Mitigation |", "|---|---|"]
        for item in rows[:10]:
            parts = [p.strip() for p in re.split(r"[:\-]", item, maxsplit=1) if p.strip()]
            if len(parts) == 2:
                risk, mitigation = parts
            else:
                risk, mitigation = item.strip(), "Mitigation to be defined during implementation"
            lines.append(f"| {risk} | {mitigation} |")
        return "\n".join(lines)

    @staticmethod
    def _feature_blocks(features: list[str]) -> str:
        clean = [f for f in features if f and f.strip()]
        if not clean:
            return "### Core Features\n- Insufficient information provided"
        lines: list[str] = []
        for idx, item in enumerate(clean[:12], 1):
            lines.append(f"### {idx}. {item[:80]}")
            lines.append(f"- {item}")
        return "\n".join(lines)

    @staticmethod
    def _user_stories(features: list[str]) -> str:
        clean = [f for f in features if f and f.strip()]
        if not clean:
            return "- As a user, I need clear decision records so I can understand context."
        lines: list[str] = []
        for idx, item in enumerate(clean[:6], 1):
            lines.append(f"- **US-{idx}:** As a user, I want {item.lower()} so the team can preserve decision context.")
        return "\n".join(lines)

    @staticmethod
    def _clean_list_items(items: list[str], max_items: int = 10) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in items:
            text = (raw or "").strip()
            if not text:
                continue
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"^[\-\*\d\.\)\s]+", "", text).strip()
            if len(text) < 5:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned or ["Insufficient information provided"]

    def _should_run_detail_pass(self, payload: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(payload.get("title") or ""),
                str(payload.get("problem_statement") or ""),
                str(payload.get("target_users") or ""),
                " ".join([str(x) for x in payload.get("features", []) if x]),
                str(payload.get("additional_notes") or ""),
            ]
        ).strip()
        # If source context is rich, run additional quality expansion calls.
        return self._estimate_tokens(text) >= 450

    async def _expand_text_section(
        self,
        stage_name: str,
        payload: dict[str, Any],
        seed_text: str,
        instruction: str,
    ) -> str:
        expanded = await self._run_text_section(
            stage_name,
            {
                "seed_text": seed_text,
                "input_context": payload,
            },
            f"{DETAIL_PROMPT} {instruction}",
        )
        return expanded if expanded else seed_text

    async def _expand_list_section(
        self,
        stage_name: str,
        payload: dict[str, Any],
        seed_items: list[str],
        instruction: str,
    ) -> list[str]:
        expanded = await self._run_list_section(
            stage_name,
            {
                "seed_items": seed_items,
                "input_context": payload,
            },
            f"{DETAIL_PROMPT} {instruction}",
        )
        return self._clean_list_items(seed_items + expanded, max_items=12)

    def merge_markdown(
        self,
        core: dict[str, Any],
        feature: dict[str, Any],
        architecture: dict[str, Any],
        nonfunctional: dict[str, Any],
        risk: dict[str, Any],
        project_title: str,
    ) -> str:
        today = datetime.now(timezone.utc).strftime("%B %d, %Y")
        stage_title = "DecisionVault — Stage 1: Decision History Core"
        features = feature.get("feature_overview", [])
        arch = architecture.get("architecture_decisions", [])
        tech = architecture.get("technical_architecture", [])
        deploy = architecture.get("deployment_strategy", [])
        nfr = nonfunctional.get("non_functional_requirements", [])
        sec = nonfunctional.get("security_compliance", [])
        risks = risk.get("risks_mitigation", [])
        constraints = risk.get("constraints", [])
        dod = risk.get("definition_of_done", [])
        scalability = risk.get("scalability_considerations", [])
        users = core.get("target_users_personas", [])
        metrics = core.get("objectives_success_metrics", [])
        out_scope = feature.get("out_of_scope", [])

        markdown = f"""# 1. Product Requirements Document (PRD)
## 1.1 {stage_title}
**Version:** 1.0
**Date:** {today}
**Status:** Draft for Development
**Owner:** Product Team
**Contributors:** Engineering, Design

## 2. Executive Summary
{core['executive_summary']}

## 3. Problem Statement
### 3.1 The Core Problem
{core['problem_statement']}

### 3.2 Why Existing Tools Fail
{self._bullets(out_scope)}

### 3.3 Success Will Mean
{self._bullets(metrics)}

## 4. Objectives and Success Metrics
### 4.1 Primary Objective
{core['executive_summary']}

### 4.2 Success Metrics (Stage 1 MVP)
{self._metrics_table(metrics)}

### 4.3 Leading Indicators (Early Validation)
{self._bullets(metrics)}

## 5. Target Users and Personas
### 5.1 Primary Persona: Sarah — Engineering Lead
{self._bullets(users[:2] or users)}

### 5.2 Secondary Persona: David — Product Manager
{self._bullets(users[2:4] or users)}

### 5.3 Tertiary Persona: Maya — Startup CTO/Founder
{self._bullets(users[4:6] or users)}

## 6. Product Scope: Stage 1 Features
### 6.1 In Scope (Must Have for MVP)
#### 6.1.1 Decision Capture System
##### Slack Thread Capture
{self._bullets(features[:2])}
##### Manual Document Upload
{self._bullets(features[2:4])}
##### Structured Decision Entry Form
{self._bullets(features[4:6])}

#### 6.1.2 Decision Record Storage
##### Immutable Decision Records
{self._bullets(features[6:7])}
##### Versioning System
{self._bullets(features[7:8])}
##### Evidence Linking
{self._bullets(features[8:9])}

#### 6.1.3 Decision Timeline View
##### Project Timeline
{self._bullets(features[9:10])}
##### Evolution Visualization
{self._bullets(features[10:11])}

#### 6.1.4 "Why Did We...?" Query System
##### Natural Language Search
{self._bullets(features[11:12])}
##### Evidence-First Results
{self._bullets(features[:1])}
##### Query Analytics (Admin View)
{self._bullets(metrics[:2])}

#### 6.1.5 Multi-Tenant Foundation
##### Tenant Architecture
{self._bullets(arch[:2])}
##### Project Hierarchy
{self._bullets(arch[2:4])}
##### User Management
{self._bullets(arch[4:5])}
##### Role-Based Access Control (RBAC)
{self._bullets(arch[5:6])}

#### 6.1.6 License and Trial System
##### Trial Mode
{self._bullets(constraints[:2])}
##### Paid Licensing
{self._bullets(constraints[2:4])}
##### Billing Integration
{self._bullets(constraints[4:5])}

### 6.2 Out of Scope (Stage 1)
{self._bullets(out_scope)}

## 7. User Stories and Use Cases
### 7.1 Epic 1: Decision Capture
#### US-1.1
{features[0] if len(features) > 0 else "Insufficient information provided"}
#### US-1.2
{features[1] if len(features) > 1 else "Insufficient information provided"}
#### US-1.3
{features[2] if len(features) > 2 else "Insufficient information provided"}

### 7.2 Epic 2: Decision Retrieval
#### US-2.1
{features[3] if len(features) > 3 else "Insufficient information provided"}
#### US-2.2
{features[4] if len(features) > 4 else "Insufficient information provided"}
#### US-2.3
{features[5] if len(features) > 5 else "Insufficient information provided"}

### 7.3 Epic 3: Decision Management
#### US-3.1
{features[6] if len(features) > 6 else "Insufficient information provided"}
#### US-3.2
{features[7] if len(features) > 7 else "Insufficient information provided"}
#### US-3.3
{features[8] if len(features) > 8 else "Insufficient information provided"}

### 7.4 Epic 4: Multi-Tenant Administration
#### US-4.1
{features[9] if len(features) > 9 else "Insufficient information provided"}
#### US-4.2
{features[10] if len(features) > 10 else "Insufficient information provided"}
#### US-4.3
{features[11] if len(features) > 11 else "Insufficient information provided"}

## 8. Technical Architecture
### 8.1 System Architecture Overview
{self._bullets(arch)}

### 8.2 Data Model (Core Entities)
#### Tenant (Organization)
{self._bullets(arch[:1])}
#### User
{self._bullets(arch[1:2] or arch[:1])}
#### Project
{self._bullets(arch[2:3] or arch[:1])}
#### Decision (Core Entity)
{self._bullets(arch[3:4] or arch[:1])}
#### Decision Relationship
{self._bullets(arch[4:5] or arch[:1])}
#### Integration (Slack)
{self._bullets(arch[5:6] or arch[:1])}

### 8.3 API Endpoints (Key Routes)
#### Authentication
{self._bullets(tech[:2])}
#### Organizations
{self._bullets(tech[2:4] or tech[:2])}
#### Projects
{self._bullets(tech[4:6] or tech[:2])}
#### Decisions
{self._bullets(tech[6:8] or tech[:2])}
#### Search
{self._bullets(tech[8:9] or tech[:2])}
#### Integrations
{self._bullets(tech[9:11] or tech[:2])}
#### Billing
{self._bullets(tech[11:12] or tech[:2])}

### 8.4 Slack Integration Architecture
#### OAuth Flow
{self._bullets(deploy[:1])}
#### Event Subscription Flow
{self._bullets(deploy[1:2] or deploy[:1])}
#### Slack Bot Permissions Required
{self._bullets(deploy[2:3] or deploy[:1])}
#### Data Privacy
{self._bullets(sec[:1] or deploy[:1])}

### 8.5 Security and Compliance
#### Data Security
{self._bullets(sec[:2] or sec)}
#### Authentication
{self._bullets(sec[2:3] or sec[:1])}
#### Access Control
{self._bullets(sec[3:4] or sec[:1])}
#### Compliance Considerations (Future)
{self._bullets(sec[4:6] or sec[:1])}

## 9. User Interface Design
### 9.1 Key Screens (Wireframe Descriptions)
#### Dashboard (Post-Login Landing)
{self._bullets(features[:1])}
#### Project Timeline View
{self._bullets(features[1:2] or features[:1])}
#### Decision Detail View
{self._bullets(features[2:3] or features[:1])}
#### Search Results View
{self._bullets(features[3:4] or features[:1])}
#### Create/Edit Decision Form
{self._bullets(features[4:5] or features[:1])}
#### Admin Settings
{self._bullets(features[5:6] or features[:1])}
#### Billing Page
{self._bullets(features[6:7] or features[:1])}

### 9.2 Design Principles
{self._bullets(features[:4])}

## 10. Dependencies and Integrations
### 10.1 External Dependencies
{self._bullets(tech[:5])}

### 10.2 API Rate Limits and Handling
{self._bullets(deploy[:3])}

### 10.3 Fallback Strategies
{self._bullets(deploy[3:6] or deploy)}

## 11. Non-Functional Requirements
### 11.1 Performance
{self._bullets(nfr[:3])}

### 11.2 Scalability
{self._bullets(scalability)}

### 11.3 Reliability
{self._bullets(nfr[3:5] or nfr)}

### 11.4 Security
{self._bullets(sec[:3])}

### 11.5 Compliance
{self._bullets(sec[3:6] or sec)}

## 12. Testing Strategy
### 12.1 Test Coverage Goals
{self._metrics_table(metrics)}

### 12.2 Key Test Scenarios
#### Decision Capture
{self._bullets(features[:2])}
#### Search and Retrieval
{self._bullets(features[2:4] or features[:2])}
#### Multi-Tenancy
{self._bullets(arch[:2])}
#### Billing
{self._bullets(constraints[:2])}
#### Performance
{self._bullets(nfr[:2])}

## 13. Launch Plan
### 13.1 Pre-Launch (Weeks -4 to -1)
{self._bullets(deploy[:3])}

### 13.2 Launch (Week 0)
{self._bullets(deploy[3:6] or deploy)}

### 13.3 Post-Launch Monitoring
{self._bullets(metrics)}

## 14. Open Questions and Assumptions
### 14.1 Open Questions
- Which assumptions require stakeholder confirmation before implementation?
- Which dependencies are highest risk for schedule variance?

### 14.2 Assumptions
- Assumptions are derived only from provided input context.
- Missing details are marked as insufficient information.

### 14.3 Risks and Mitigations
{self._risk_table(risks)}

## 15. Success Criteria and Definition of Done
### 15.1 Stage 1 is "Done" When
{self._bullets(dod)}

### 15.2 Stage 1 is "Successful" When (6-Month Post-Launch)
{self._bullets(metrics)}

## 16. Appendix
### 16.1 Glossary
- Decision Record: structured, evidence-linked record of a team decision.
- Evidence Link: original source artifact URL (message/thread/document).
- Tenant: organization-level isolated workspace.

### 16.2 References
- [1] User-provided requirement context and structured inputs.
- [2] DecisionVault internal product notes supplied in this session.

## 17. Document Change Log
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | {today} | Product Team | Initial generated PRD draft |
"""

        missing = [h for h in REQUIRED_SECTION_HEADINGS if h not in markdown]
        if missing:
            raise ValueError(f"Missing required sections: {missing}")

        self.sections_generated = REQUIRED_SECTION_HEADINGS.copy()
        return markdown


async def generate_multistep_prd(
    payload: PRDGenerateRequest,
    tenant_id: str,
    progress_cb: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> PRDMultiStepResponse:
    orchestrator = PRDOrchestrator(tenant_id=tenant_id, progress_cb=progress_cb)

    shared_payload = {
        "title": payload.title,
        "problem_statement": payload.problem_statement,
        "target_users": payload.target_users,
        "features": payload.features,
        "additional_notes": payload.additional_notes or "",
    }

    core = await orchestrator.generate_core_sections(shared_payload)
    feature = await orchestrator.generate_feature_sections(shared_payload)
    architecture = await orchestrator.generate_architecture_sections(shared_payload)
    nonfunctional = await orchestrator.generate_nonfunctional_sections(shared_payload)
    risk = await orchestrator.generate_risk_sections(shared_payload)

    markdown = orchestrator.merge_markdown(
        core,
        feature,
        architecture,
        nonfunctional,
        risk,
        project_title=payload.title,
    )

    return PRDMultiStepResponse(
        status="success",
        pages_estimated=5,
        sections_generated=orchestrator.sections_generated,
        total_tokens_used=orchestrator.total_tokens_used,
        prd_markdown=markdown,
        required_sections=REQUIRED_SECTION_HEADINGS,
        missing_sections=[],
        has_all_required_sections=True,
    )
