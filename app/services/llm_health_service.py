from __future__ import annotations

import time

from app.core.config import settings


async def probe_llm() -> dict:
    if not settings.llm_api_key:
        return {"status": "misconfigured", "reason": "missing_api_key"}
    if not settings.llm_model:
        return {"status": "misconfigured", "reason": "missing_model"}

    prompt = "Ping."
    started = time.perf_counter()
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.llm_api_key,
        )
        await llm.ainvoke(prompt)
    elif settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            return {"status": "misconfigured", "reason": "gemini_dependencies_missing", "error": str(exc)}
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.0,
            google_api_key=settings.llm_api_key,
        )
        await llm.ainvoke(prompt)
    else:
        return {"status": "misconfigured", "reason": "unsupported_provider"}
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model, "latency_ms": elapsed_ms}
