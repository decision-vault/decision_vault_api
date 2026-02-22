from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TypedDict
import httpx

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.schemas.prd_generation import PRDStructuredOutput
from app.services.cache_service import build_cache_key, cache_get, cache_set
from app.services.llm_usage_service import log_llm_usage
from app.services.token_limiter import TokenBudget, TokenLimiter

logger = logging.getLogger("decisionvault.prd")

MAX_OUTPUT_WORDS = 4000
PRD_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

STRICT_SYSTEM_PROMPT = (
    "You are a senior product architect. Expand and structure the user's idea into a professional PRD. "
    "Only refine and clarify the provided input. "
    "Hard constraints: No invented statistics. No invented integrations. No invented features. "
    "Only use provided input. If required data is missing, write exactly: Insufficient information provided."
)


class PRDGraphState(TypedDict):
    tenant_id: str
    title: str
    problem_statement: str
    target_users: str
    features: list[str]
    additional_notes: str | None
    normalized_features_block: str
    validation_error: str | None
    structured_output: PRDStructuredOutput | None
    enhanced_output: PRDStructuredOutput | None
    prd_markdown: str
    sections_generated: list[str]
    confidence_score: float
    token_usage_total: int
    parse_retry_attempts: int
    hallucination_retry_attempts: int
    hallucination_detected: bool


def _collapse_ws(text: str) -> str:
    return " ".join(text.split()).strip()


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _clip_words(text: str, max_words: int = MAX_OUTPUT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _extract_token_usage(message) -> int:
    usage = getattr(message, "response_metadata", {}) or {}
    token_usage = usage.get("token_usage") or usage.get("usage") or {}
    return int(token_usage.get("total_tokens") or 0)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_block(text: str) -> str:
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return cleaned
    return cleaned[start : end + 1]


def _collect_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = _collapse_ws(value)
        return [v] if v else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_strings(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        # Prefer common content-bearing keys first.
        for key in ("content", "title", "text", "description", "value"):
            if key in value:
                parts.extend(_collect_strings(value.get(key)))
        # Then recurse remaining keys.
        for k, v in value.items():
            if k in {"content", "title", "text", "description", "value"}:
                continue
            parts.extend(_collect_strings(v))
        return parts
    return []


def _to_list_str(value) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for item in _collect_strings(value):
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _coerce_structured_output(raw_text: str) -> PRDStructuredOutput:
    data = json.loads(_extract_json_block(raw_text))
    if not isinstance(data, dict):
        raise ValueError("Structured output must be a JSON object")

    exec_summary = _to_list_str(data.get("executive_summary"))
    problem_stmt = _to_list_str(data.get("problem_statement"))
    target_users = _to_list_str(data.get("target_users"))
    feature_overview = _to_list_str(data.get("feature_overview"))
    technical_considerations = _to_list_str(data.get("technical_considerations"))
    risks = _to_list_str(data.get("risks"))
    success_metrics = _to_list_str(data.get("success_metrics"))

    normalized = {
        "executive_summary": exec_summary[0] if exec_summary else "Insufficient information provided",
        "problem_statement": problem_stmt[0] if problem_stmt else "Insufficient information provided",
        "target_users": target_users or ["Insufficient information provided"],
        "feature_overview": feature_overview or ["Insufficient information provided"],
        "technical_considerations": technical_considerations or ["Insufficient information provided"],
        "risks": risks or ["Insufficient information provided"],
        "success_metrics": success_metrics or ["Insufficient information provided"],
    }
    return PRDStructuredOutput.model_validate(normalized)


def _llm(model_name: str, max_tokens: int) -> ChatOpenAI:
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "huggingface":
        api_key = settings.hf_api_token
        base_url = settings.hf_router_base_url
        resolved_model = settings.hf_openai_model or model_name
    elif provider == "lmstudio":
        api_key = settings.llm_api_key or "lm-studio"
        base_url = settings.lmstudio_base_url
        resolved_model = settings.lmstudio_model or model_name
    else:
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        resolved_model = model_name

    if not api_key:
        raise ValueError("LLM API key not configured")

    logger.warning(
        "prd_llm_selected provider=%s model=%s base_url=%s max_tokens=%s",
        provider or "default",
        resolved_model,
        base_url,
        max_tokens,
    )

    kwargs = {
        "model": resolved_model,
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _extract_lmstudio_text(payload: dict) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            return "\n".join(parts).strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


async def _invoke_lmstudio_chat(prompt_text: str, output_tokens: int) -> tuple[str, int]:
    model = settings.lmstudio_model or settings.llm_model
    base_url = (settings.lmstudio_base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("LM Studio base URL is not configured")
    url = f"{base_url}{settings.lmstudio_chat_path}"
    request_body = {
        "model": model,
        "input": prompt_text,
        "temperature": 0.2,
        "context_length": max(4000, settings.llm_max_input_tokens + output_tokens),
    }
    logger.warning(
        "prd_lmstudio_request base_url=%s chat_path=%s model=%s output_tokens=%s",
        base_url,
        settings.lmstudio_chat_path,
        model,
        output_tokens,
    )
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, json=request_body, headers={"Authorization": f"Bearer {settings.llm_api_key or 'lm-studio'}"})
        response.raise_for_status()
        payload = response.json()
    text = _extract_lmstudio_text(payload)
    if not text:
        raise ValueError(f"LM Studio response did not contain text output: {payload}")
    stats = payload.get("stats") if isinstance(payload, dict) else {}
    if isinstance(stats, dict):
        usage_tokens = int((stats.get("input_tokens") or 0) + (stats.get("total_output_tokens") or 0))
    else:
        usage_tokens = _estimate_tokens(prompt_text + "\n" + text)
    logger.warning(
        "prd_lmstudio_response model=%s usage_tokens=%s output_chars=%s",
        model,
        usage_tokens,
        len(text),
    )
    return text, usage_tokens


def _to_markdown(output: PRDStructuredOutput) -> str:
    md = f"""# Product Requirements Document (PRD)

## Executive Summary
{output.executive_summary}

## Problem Statement
{output.problem_statement}

## Target Users
{chr(10).join([f"- {item}" for item in output.target_users])}

## Feature Overview
{chr(10).join([f"- {item}" for item in output.feature_overview])}

## Technical Considerations
{chr(10).join([f"- {item}" for item in output.technical_considerations])}

## Risks
{chr(10).join([f"- {item}" for item in output.risks])}

## Success Metrics
{chr(10).join([f"- {item}" for item in output.success_metrics])}
"""
    return _clip_words(md, MAX_OUTPUT_WORDS)


def detect_hallucination(user_input: str, prd_output: str) -> bool:
    input_text = user_input.lower()
    output_text = prd_output.lower()

    input_numbers = set(re.findall(r"\d+(?:[\.,]\d+)?", input_text))
    output_numbers = set(re.findall(r"\d+(?:[\.,]\d+)?", output_text))
    if output_numbers - input_numbers:
        return True

    known_integrations = {
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
    output_integrations = {name for name in known_integrations if name in output_text}
    input_integrations = {name for name in known_integrations if name in input_text}
    if output_integrations - input_integrations:
        return True

    if "%" in output_text and "%" not in input_text:
        return True
    if re.search(r"[$€£₹]", output_text) and not re.search(r"[$€£₹]", input_text):
        return True
    return False


async def _invoke_structured(
    *,
    state: PRDGraphState,
    model_name: str,
    stronger_instruction: str = "",
) -> tuple[PRDStructuredOutput, int, int]:
    parser = PydanticOutputParser(pydantic_object=PRDStructuredOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRICT_SYSTEM_PROMPT + (f" {stronger_instruction}" if stronger_instruction else "")),
            (
                "user",
                "Title: {title}\n"
                "Problem Statement: {problem_statement}\n"
                "Target Users: {target_users}\n"
                "Features:\n{features_block}\n"
                "Additional Notes: {additional_notes}\n"
                "Existing Draft (if any):\n{existing_draft}\n\n"
                "Return valid JSON only.\n{format_instructions}",
            ),
        ]
    )

    input_block = "\n".join(
        [
            state["title"],
            state["problem_statement"],
            state["target_users"],
            state["normalized_features_block"],
            state.get("additional_notes") or "",
        ]
    )
    input_tokens = _estimate_tokens(input_block)
    output_tokens = min(settings.llm_max_output_tokens, 1800)
    limiter = TokenLimiter(TokenBudget(settings.llm_max_input_tokens, settings.llm_max_output_tokens))
    limiter.enforce(input_tokens=input_tokens, output_tokens=output_tokens)

    parse_retries = 0
    last_err = None
    existing = state.get("enhanced_output") or state.get("structured_output")
    existing_draft = existing.model_dump_json(indent=2) if existing else ""
    provider = (settings.llm_provider or "").strip().lower()

    llm = None
    chain = None
    if provider != "lmstudio":
        llm = _llm(model_name, output_tokens)
        chain = prompt | llm

    for attempt in range(2):
        if provider == "lmstudio":
            prompt_text = (
                f"{STRICT_SYSTEM_PROMPT}{(' ' + stronger_instruction) if stronger_instruction else ''}\n\n"
                f"Title: {state['title']}\n"
                f"Problem Statement: {state['problem_statement']}\n"
                f"Target Users: {state['target_users']}\n"
                f"Features:\n{state['normalized_features_block']}\n"
                f"Additional Notes: {state.get('additional_notes') or 'Insufficient information provided'}\n"
                f"Existing Draft (if any):\n{existing_draft}\n\n"
                f"Return valid JSON only.\n{parser.get_format_instructions()}\n"
            )
            content, usage_tokens = await _invoke_lmstudio_chat(prompt_text, output_tokens)
        else:
            message = await chain.ainvoke(
                {
                    "title": state["title"],
                    "problem_statement": state["problem_statement"],
                    "target_users": state["target_users"],
                    "features_block": state["normalized_features_block"],
                    "additional_notes": state.get("additional_notes") or "Insufficient information provided",
                    "existing_draft": existing_draft,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            content = (getattr(message, "content", "") or "").strip()
            usage_tokens = _extract_token_usage(message)
        try:
            await log_llm_usage(
                feature="prd_generate",
                model=settings.lmstudio_model if provider == "lmstudio" else model_name,
                input_tokens=input_tokens,
                output_tokens=max(1, usage_tokens - input_tokens),
                tenant_id=state["tenant_id"],
            )
        except Exception:
            logger.warning("llm_usage_log_skipped")
        try:
            return parser.parse(content), usage_tokens, parse_retries
        except Exception as exc:
            # LM Studio can return malformed-but-recoverable JSON-like payloads.
            try:
                coerced = _coerce_structured_output(content)
                logger.warning("prd_parse_coerced attempt=%s", attempt + 1)
                return coerced, usage_tokens, parse_retries
            except Exception as coerce_exc:
                last_err = coerce_exc
                parse_retries += 1
                logger.warning(
                    "prd_parse_retry attempt=%s parse_error=%s coerce_error=%s",
                    attempt + 1,
                    str(exc),
                    str(coerce_exc),
                )

    raise ValueError(f"Failed to parse structured PRD output: {last_err}")


async def normalize_input(state: PRDGraphState) -> PRDGraphState:
    features = [_collapse_ws(item) for item in state["features"] if _collapse_ws(item)]
    features_block = "\n".join([f"- {item}" for item in features])
    logger.info("prd_input", extra={"input_size": len(features_block), "feature_count": len(features)})
    return {
        **state,
        "title": _collapse_ws(state["title"]),
        "problem_statement": _collapse_ws(state["problem_statement"]),
        "target_users": _collapse_ws(state["target_users"]),
        "additional_notes": _collapse_ws(state.get("additional_notes") or ""),
        "features": features,
        "normalized_features_block": features_block,
    }


async def validate_product_intent(state: PRDGraphState) -> PRDGraphState:
    vague_terms = ["something", "anything", "tbd", "later", "etc", "maybe", "not sure"]
    text = f"{state['problem_statement']} {state['target_users']} {' '.join(state['features'])}".lower()
    if any(term in text for term in vague_terms):
        return {**state, "validation_error": "Vague product intent. Add specific details."}
    return {**state, "validation_error": None}


async def generate_structured_prd(state: PRDGraphState) -> PRDGraphState:
    if state.get("validation_error"):
        return state
    parsed, used_tokens, parse_retries = await _invoke_structured(
        state=state,
        model_name=settings.llm_model or settings.llm_strong_model,
    )
    return {
        **state,
        "structured_output": parsed,
        "token_usage_total": state.get("token_usage_total", 0) + used_tokens,
        "parse_retry_attempts": state.get("parse_retry_attempts", 0) + parse_retries,
    }


async def enhance_prd(state: PRDGraphState) -> PRDGraphState:
    if state.get("validation_error"):
        return state
    parsed, used_tokens, parse_retries = await _invoke_structured(
        state=state,
        model_name=settings.llm_model or settings.llm_cheap_model,
    )
    return {
        **state,
        "enhanced_output": parsed,
        "token_usage_total": state.get("token_usage_total", 0) + used_tokens,
        "parse_retry_attempts": state.get("parse_retry_attempts", 0) + parse_retries,
    }


async def hallucination_check(state: PRDGraphState) -> PRDGraphState:
    output_obj = state.get("enhanced_output") or state.get("structured_output")
    if not output_obj:
        return {**state, "validation_error": "No PRD output generated"}
    user_input = "\n".join(
        [state["title"], state["problem_statement"], state["target_users"], state.get("additional_notes") or "", "\n".join(state["features"])]
    )
    output_text = output_obj.model_dump_json()
    flagged = detect_hallucination(user_input, output_text)
    if flagged:
        logger.warning("prd_hallucination_triggered", extra={"attempt": state.get("hallucination_retry_attempts", 0)})
    return {**state, "hallucination_detected": flagged}


async def enhance_prd_strict(state: PRDGraphState) -> PRDGraphState:
    if not state.get("hallucination_detected"):
        return state
    parsed, used_tokens, parse_retries = await _invoke_structured(
        state=state,
        model_name=settings.llm_model or settings.llm_cheap_model,
        stronger_instruction="You introduced unsupported information. Remove all invented details.",
    )
    return {
        **state,
        "enhanced_output": parsed,
        "hallucination_detected": False,
        "hallucination_retry_attempts": state.get("hallucination_retry_attempts", 0) + 1,
        "token_usage_total": state.get("token_usage_total", 0) + used_tokens,
        "parse_retry_attempts": state.get("parse_retry_attempts", 0) + parse_retries,
    }


async def format_markdown(state: PRDGraphState) -> PRDGraphState:
    if state.get("validation_error"):
        return state
    output_obj = state.get("enhanced_output") or state.get("structured_output")
    if not output_obj:
        return {**state, "validation_error": "No PRD output generated"}

    md = _to_markdown(output_obj)
    sections_generated = re.findall(r"^##\s+(.+)$", md, flags=re.MULTILINE)

    confidence = 0.78
    if len(state["features"]) >= 5:
        confidence += 0.08
    if state.get("additional_notes"):
        confidence += 0.06
    if state.get("hallucination_retry_attempts", 0) > 0:
        confidence -= 0.08
    confidence = max(0.0, min(0.99, confidence))

    logger.info(
        "prd_generate_usage",
        extra={
            "input_size": len((state["title"] + state["problem_statement"]).encode("utf-8")),
            "output_size": len(md.encode("utf-8")),
            "retry_attempts": state.get("hallucination_retry_attempts", 0),
            "hallucination_triggered": state.get("hallucination_retry_attempts", 0) > 0,
        },
    )

    return {
        **state,
        "prd_markdown": md,
        "sections_generated": sections_generated,
        "confidence_score": round(confidence, 2),
    }


def _route_after_validation(state: PRDGraphState) -> str:
    return "END" if state.get("validation_error") else "GenerateStructuredPRD"


def _route_after_hallucination(state: PRDGraphState) -> str:
    if state.get("hallucination_detected") and state.get("hallucination_retry_attempts", 0) < 1:
        return "EnhancePRDStrict"
    if state.get("hallucination_detected") and state.get("hallucination_retry_attempts", 0) >= 1:
        return "END"
    return "FormatMarkdown"


def build_prd_graph():
    graph = StateGraph(PRDGraphState)
    graph.add_node("NormalizeInput", normalize_input)
    graph.add_node("ValidateProductIntent", validate_product_intent)
    graph.add_node("GenerateStructuredPRD", generate_structured_prd)
    graph.add_node("EnhancePRD", enhance_prd)
    graph.add_node("HallucinationCheck", hallucination_check)
    graph.add_node("EnhancePRDStrict", enhance_prd_strict)
    graph.add_node("FormatMarkdown", format_markdown)

    graph.set_entry_point("NormalizeInput")
    graph.add_edge("NormalizeInput", "ValidateProductIntent")
    graph.add_conditional_edges(
        "ValidateProductIntent",
        _route_after_validation,
        {"GenerateStructuredPRD": "GenerateStructuredPRD", "END": END},
    )
    graph.add_edge("GenerateStructuredPRD", "EnhancePRD")
    graph.add_edge("EnhancePRD", "HallucinationCheck")
    graph.add_conditional_edges(
        "HallucinationCheck",
        _route_after_hallucination,
        {"EnhancePRDStrict": "EnhancePRDStrict", "FormatMarkdown": "FormatMarkdown", "END": END},
    )
    graph.add_edge("EnhancePRDStrict", "FormatMarkdown")
    graph.add_edge("FormatMarkdown", END)
    return graph.compile()


async def run_prd_pipeline(payload: dict, timeout_seconds: int = 60) -> dict:
    provider = (settings.llm_provider or "default").strip().lower()
    resolved_model = (
        settings.lmstudio_model if provider == "lmstudio"
        else settings.hf_openai_model if provider == "huggingface"
        else settings.llm_model
    )
    logger.warning(
        "prd_pipeline_start provider=%s model=%s",
        provider,
        resolved_model,
    )
    normalized_input = "\n".join(
        [
            f"provider={provider}",
            f"model={resolved_model}",
            payload["title"].strip(),
            payload["problem_statement"].strip(),
            payload["target_users"].strip(),
            "\n".join([item.strip() for item in payload["features"]]),
            (payload.get("additional_notes") or "").strip(),
        ]
    )
    cache_key = build_cache_key(
        feature="prd_generate",
        tenant_id=payload["tenant_id"],
        normalized_input=normalized_input,
    )
    if provider != "lmstudio":
        cached = await cache_get(cache_key)
        if cached:
            logger.warning("prd_pipeline_cache_hit provider=%s model=%s", provider, resolved_model)
            return cached

    graph = build_prd_graph()
    initial_state: PRDGraphState = {
        "tenant_id": payload["tenant_id"],
        "title": payload["title"],
        "problem_statement": payload["problem_statement"],
        "target_users": payload["target_users"],
        "features": payload["features"],
        "additional_notes": payload.get("additional_notes"),
        "normalized_features_block": "",
        "validation_error": None,
        "structured_output": None,
        "enhanced_output": None,
        "prd_markdown": "",
        "sections_generated": [],
        "confidence_score": 0.0,
        "token_usage_total": 0,
        "parse_retry_attempts": 0,
        "hallucination_retry_attempts": 0,
        "hallucination_detected": False,
    }
    result = await asyncio.wait_for(graph.ainvoke(initial_state), timeout=timeout_seconds)
    if result.get("validation_error"):
        raise ValueError(result["validation_error"])
    if result.get("hallucination_detected"):
        raise ValueError("Hallucination suspected")

    output = {
        "prd_markdown": result.get("prd_markdown", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "sections_generated": result.get("sections_generated", []),
    }
    if provider != "lmstudio":
        await cache_set(cache_key, output, PRD_CACHE_TTL_SECONDS)
    return output
