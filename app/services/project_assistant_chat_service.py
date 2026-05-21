from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.token_limiter import TokenLimiter
from app.services.project_vector_memory_service import (
    retrieve_project_knowledge_chunks,
    sync_project_knowledge_chunks,
)

logger = logging.getLogger("decisionvault.project_assistant_chat")


def _provider_config() -> tuple[str, str | None, str | None, str]:
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


def _trim_text(value: str, max_chars: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_chat_history(messages: list[dict[str, Any]], max_items: int = 18) -> str:
    items: list[str] = []
    for m in (messages or [])[-max_items:]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").lower()
        text = str(m.get("text") or "").strip()
        kind = str(m.get("kind") or "").strip()
        if not text:
            continue
        if kind in {"status"}:
            continue
        if role not in {"user", "assistant"}:
            continue
        speaker = "User" if role == "user" else "Assistant"
        items.append(f"{speaker}: {_trim_text(text, 700)}")
    return "\n".join(items).strip()


async def generate_project_assistant_reply(
    *,
    tenant_id: str,
    project_id: str,
    structured: dict[str, Any],
    chat_messages: list[dict[str, Any]],
    user_message: str,
) -> str:
    model_name, api_key, base_url, provider = _provider_config()
    if not api_key:
        raise ValueError("LLM API key not configured")

    retrieved_chunks: list[str] = []
    try:
        await sync_project_knowledge_chunks(tenant_id=tenant_id, project_id=project_id)
        retrieval_query = "\n".join(
            [
                f"user_message: {user_message}",
                f"project_name: {structured.get('project_name')}",
                f"problem_statement: {structured.get('problem_statement')}",
                "target_users: " + ", ".join(str(x) for x in (structured.get("target_users") or [])),
                "features: " + ", ".join(str(x) for x in (structured.get("desired_features") or [])),
            ]
        )
        retrieved_chunks = await retrieve_project_knowledge_chunks(
            tenant_id=tenant_id,
            project_id=project_id,
            query_text=retrieval_query,
            top_k=6,
        )
    except Exception:
        retrieved_chunks = []

    # LM Studio / llama.cpp will error if prompt exceeds model context length.
    # We keep this prompt within a conservative budget, trimming context dynamically.
    context_window = int(getattr(settings, "llm_context_window_tokens", 4096) or 4096)
    if provider == "lmstudio":
        context_window = min(context_window, 4096)
    safety = int(getattr(settings, "llm_context_safety_margin_tokens", 256) or 256)
    max_total_tokens = max(512, context_window - safety)

    desired_output_tokens = 500
    base_system = (
        "You are DecisionVault Assistant, an AI agent helping users build product documents and make decisions.\n"
        "Respond concisely and practically. If the user asks for output that should be captured as a document section, "
        "propose a short, structured draft.\n"
        "Do not invent technologies or facts not present in the context.\n\n"
    )

    structured_max_chars = 4500
    history_max_items = 14
    chunk_top_k = min(6, len(retrieved_chunks or []))
    chunk_max_chars = 650

    def build_prompt() -> str:
        structured_json = _trim_text(json.dumps(structured or {}, ensure_ascii=False, indent=2), structured_max_chars)
        history = _render_chat_history(chat_messages, max_items=history_max_items)
        chunks_block = "\n".join(
            [f"- {_trim_text(str(c), chunk_max_chars)}" for c in (retrieved_chunks or [])[:chunk_top_k]]
        ).strip()

        p = base_system
        p += "PROJECT CONTEXT (structured requirements JSON):\n" + structured_json + "\n\n"
        if chunks_block:
            p += "RELATED PROJECT KNOWLEDGE (retrieved snippets):\n" + chunks_block + "\n\n"
        if history:
            p += "CHAT HISTORY:\n" + history + "\n\n"
        p += f"USER MESSAGE:\n{_trim_text(user_message, 1800)}\n\nASSISTANT RESPONSE:\n"
        return p

    prompt = build_prompt()
    for _ in range(10):
        prompt_tokens = TokenLimiter.estimate_tokens(prompt)
        if prompt_tokens + desired_output_tokens <= max_total_tokens:
            break
        if history_max_items > 6:
            history_max_items = max(6, history_max_items - 2)
        elif chunk_top_k > 2:
            chunk_top_k = max(2, chunk_top_k - 1)
        elif structured_max_chars > 2000:
            structured_max_chars = max(2000, structured_max_chars - 1000)
        elif chunk_max_chars > 300:
            chunk_max_chars = max(300, chunk_max_chars - 150)
        else:
            # Minimal fallback: no JSON dump, no retrieval, no history.
            prompt = (
                base_system
                + "PROJECT CONTEXT (short):\n"
                + _trim_text(
                    "\n".join(
                        [
                            f"project_name: {structured.get('project_name')}",
                            f"problem_statement: {structured.get('problem_statement')}",
                            "target_users: " + ", ".join(str(x) for x in (structured.get("target_users") or [])),
                            "features: " + ", ".join(str(x) for x in (structured.get("desired_features") or [])),
                        ]
                    ),
                    1600,
                )
                + "\n\nUSER MESSAGE:\n"
                + _trim_text(user_message, 1800)
                + "\n\nASSISTANT RESPONSE:\n"
            )
            break
        prompt = build_prompt()

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        top_p=0.8,
        max_tokens=desired_output_tokens,
        api_key=api_key,
        base_url=_normalize_openai_base_url(base_url, provider),
    )
    msg = await llm.ainvoke(prompt)
    text = (getattr(msg, "content", "") or "").strip()
    return _trim_text(text or "Insufficient information provided.", 5000)
