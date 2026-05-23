from __future__ import annotations

import time
import httpx

from app.core.config import settings


async def probe_llm() -> dict:
    if not settings.llm_api_key:
        return {"status": "misconfigured", "reason": "missing_api_key"}
    if not settings.llm_model:
        return {"status": "misconfigured", "reason": "missing_model"}

    prompt = "Ping."
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            if settings.llm_provider == "openai":
                headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
                data = {
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                }
                url = settings.llm_base_url or "https://api.openai.com/v1/chat/completions"
                resp = await client.post(url, json=data, headers=headers, timeout=10.0)
                resp.raise_for_status()
            elif settings.llm_provider == "gemini":
                base_url = settings.llm_base_url or "https://generativelanguage.googleapis.com/v1beta"
                if "openai" in base_url:
                    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
                    data = {
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    }
                    resp = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=data, headers=headers, timeout=10.0)
                else:
                    url = f"{base_url.rstrip('/')}/models/{settings.llm_model}:generateContent?key={settings.llm_api_key}"
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}]
                    }
                    resp = await client.post(url, json=data, timeout=10.0)
                resp.raise_for_status()
            else:
                return {"status": "misconfigured", "reason": "unsupported_provider"}
    except Exception as exc:
        return {"status": "error", "provider": settings.llm_provider, "model": settings.llm_model, "error": str(exc)}

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model, "latency_ms": elapsed_ms}
