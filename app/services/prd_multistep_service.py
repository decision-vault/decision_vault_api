from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Type

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
    "Write concrete, professional content and avoid placeholder wording."
)

DETAIL_PROMPT = (
    "Expand depth and clarity while preserving meaning. "
    "Do not introduce new facts not present in input. "
    "Prefer implementation-ready wording."
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
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
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
                return TextSection(content=parsed_json["content"].strip())
            strings = cls._collect_strings(parsed_json)
            return TextSection(content=(strings[0] if strings else "Insufficient information provided"))

        if schema is ListSection:
            if isinstance(parsed_json, dict) and isinstance(parsed_json.get("items"), list):
                items = [str(item).strip() for item in parsed_json["items"] if str(item).strip()]
                return ListSection(items=items or ["Insufficient information provided"])
            items = cls._collect_strings(parsed_json)
            deduped: list[str] = []
            seen: set[str] = set()
            for item in items:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
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
                raw_json = json.loads(self._extract_json_block(raw_text))
                parsed = self._coerce_stage_output(schema, raw_json)
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
                return StageRunResult(parsed, input_tokens, used_tokens, retry_count)
            except Exception as exc:
                last_error = exc
                retry_count = 1

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
                "Write Executive Summary in 2 short paragraphs: product purpose and user value.",
            ),
            "problem_statement": await self._run_text_section(
                "stage_2_problem_statement",
                {"problem_statement": payload["problem_statement"], "additional_notes": payload.get("additional_notes", "")},
                "Write a clear Problem Statement in 4-6 sentences with scope and pain points.",
            ),
            "target_users_personas": await self._run_list_section(
                "stage_3_target_users_personas",
                {"target_users": payload["target_users"], "problem_statement": payload["problem_statement"]},
                "List 4-8 Target Users & Personas as concise business-oriented bullets.",
            ),
            "objectives_success_metrics": await self._run_list_section(
                "stage_4_objectives_success_metrics",
                {"features": payload["features"], "additional_notes": payload.get("additional_notes", "")},
                "List 5-8 Objectives & Success Metrics, each measurable when possible.",
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
                "List 6-10 Feature Overview items as implementation-ready bullets.",
            ),
            "out_of_scope": await self._run_list_section(
                "stage_6_out_of_scope",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List Out of Scope items only when explicitly implied by input.",
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
                "List 5-10 Architecture Decisions based on provided stack/features only.",
            ),
            "technical_architecture": await self._run_list_section(
                "stage_8_technical_architecture",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List 6-10 Technical Architecture details (components, data flow, integration boundaries).",
            ),
            "deployment_strategy": await self._run_list_section(
                "stage_9_deployment_strategy",
                {"additional_notes": payload.get("additional_notes", "")},
                "List 4-8 Deployment Strategy details (environment, rollout, observability hooks).",
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
                "List 6-10 Non-Functional Requirements with concrete quality expectations.",
            ),
            "security_compliance": await self._run_list_section(
                "stage_11_security_compliance",
                {"additional_notes": payload.get("additional_notes", "")},
                "List 4-8 Security & Compliance requirements from provided context only.",
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
                "List 4-8 Scalability Considerations aligned with expected usage.",
            ),
            "risks_mitigation": await self._run_list_section(
                "stage_13_risks_mitigation",
                {"features": payload["features"], "additional_notes": payload.get("additional_notes", "")},
                "List 5-10 Risks & Mitigation items as paired actionable bullets.",
            ),
            "constraints": await self._run_list_section(
                "stage_14_constraints",
                {"additional_notes": payload.get("additional_notes", ""), "features": payload["features"]},
                "List 4-8 delivery/business/technical Constraints.",
            ),
            "definition_of_done": await self._run_list_section(
                "stage_15_definition_of_done",
                {"features": payload["features"], "problem_statement": payload["problem_statement"]},
                "List 6-10 Definition of Done criteria that are testable and release-ready.",
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
        markdown = f"""# Product Requirements Document (PRD)
{project_title} — Version 1.0
Date: {today}
Status: Draft for Development

## 1. Executive Summary
{core['executive_summary']}

## 2. Problem Statement
{core['problem_statement']}

## 3. Target Users & Personas
{self._bullets(core['target_users_personas'])}

## 4. Objectives & Success Metrics
{self._bullets(core['objectives_success_metrics'])}

## 5. Feature Overview (Stage 1)
{self._bullets(feature['feature_overview'])}

## 6. Out of Scope
{self._bullets(feature['out_of_scope'])}

## 7. Architecture Decisions
{self._bullets(architecture['architecture_decisions'])}

## 8. Technical Architecture
{self._bullets(architecture['technical_architecture'])}

## 9. Deployment Strategy
{self._bullets(architecture['deployment_strategy'])}

## 10. Non-Functional Requirements
{self._bullets(nonfunctional['non_functional_requirements'])}

## 11. Security & Compliance
{self._bullets(nonfunctional['security_compliance'])}

## 12. Scalability Considerations
{self._bullets(risk['scalability_considerations'])}

## 13. Risks & Mitigation
{self._bullets(risk['risks_mitigation'])}

## 14. Constraints
{self._bullets(risk['constraints'])}

## 15. Definition of Done
{self._bullets(risk['definition_of_done'])}

---END OF PRD---
"""

        required_headings = [
            "## 1. Executive Summary",
            "## 2. Problem Statement",
            "## 3. Target Users & Personas",
            "## 4. Objectives & Success Metrics",
            "## 5. Feature Overview (Stage 1)",
            "## 6. Out of Scope",
            "## 7. Architecture Decisions",
            "## 8. Technical Architecture",
            "## 9. Deployment Strategy",
            "## 10. Non-Functional Requirements",
            "## 11. Security & Compliance",
            "## 12. Scalability Considerations",
            "## 13. Risks & Mitigation",
            "## 14. Constraints",
            "## 15. Definition of Done",
        ]
        missing = [h for h in required_headings if h not in markdown]
        if missing:
            raise ValueError(f"Missing required sections: {missing}")

        self.sections_generated = [
            "Executive Summary",
            "Problem Statement",
            "Target Users & Personas",
            "Objectives & Success Metrics",
            "Feature Overview (Stage 1)",
            "Out of Scope",
            "Architecture Decisions",
            "Technical Architecture",
            "Deployment Strategy",
            "Non-Functional Requirements",
            "Security & Compliance",
            "Scalability Considerations",
            "Risks & Mitigation",
            "Constraints",
            "Definition of Done",
        ]
        return markdown


async def generate_multistep_prd(payload: PRDGenerateRequest, tenant_id: str) -> PRDMultiStepResponse:
    orchestrator = PRDOrchestrator(tenant_id=tenant_id)

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
    )
