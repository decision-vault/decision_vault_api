from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from transformers import AutoTokenizer
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger("decisionvault.hf")


@dataclass
class RemoteModel:
    client: OpenAI
    tokenizer: AutoTokenizer | None
    model_name: str
    mode: str


@dataclass
class LocalModel:
    model: object
    tokenizer: AutoTokenizer
    model_name: str


_model_lock = asyncio.Lock()
_model: RemoteModel | LocalModel | None = None


async def load_remote_model() -> RemoteModel:
    if not settings.hf_api_token:
        raise RuntimeError("HF_API_TOKEN is required for remote mode")
    if settings.remote_provider == "hf_router":
        if not settings.hf_openai_model:
            raise RuntimeError("HF_OPENAI_MODEL is required for remote mode")
        client = OpenAI(
            base_url=settings.hf_router_base_url,
            api_key=settings.hf_api_token,
        )
        model_name = settings.hf_openai_model
        mode = "hf_router"
    elif settings.remote_provider == "lmstudio":
        if not settings.lmstudio_model:
            raise RuntimeError("LMSTUDIO_MODEL is required for remote mode")
        client = OpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.hf_api_token,
        )
        model_name = settings.lmstudio_model
        mode = "lmstudio"
    else:
        raise RuntimeError("REMOTE_PROVIDER must be 'hf_router' or 'lmstudio'")
    tokenizer = None
    if settings.hf_tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.hf_tokenizer_name)
        except Exception:
            logger.warning("hf_tokenizer_load_failed", extra={"model": settings.hf_tokenizer_name})
    return RemoteModel(
        client=client,
        tokenizer=tokenizer,
        model_name=model_name,
        mode=mode,
    )


async def load_local_model() -> LocalModel:
    if not settings.hf_model_name and not settings.local_model_path:
        raise RuntimeError("HF_MODEL_NAME or LOCAL_MODEL_PATH is required for local mode")
    model_id = settings.local_model_path or settings.hf_model_name
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError("Local mode requires torch and transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if not settings.local_device else None,
    )
    if settings.local_device:
        model.to(settings.local_device)
    model.eval()
    return LocalModel(model=model, tokenizer=tokenizer, model_name=model_id)


async def get_model() -> RemoteModel | LocalModel:
    global _model
    async with _model_lock:
        if _model is not None:
            return _model
        if settings.model_run_mode == "remote":
            _model = await load_remote_model()
        elif settings.model_run_mode == "local":
            _model = await load_local_model()
        else:
            raise RuntimeError("MODEL_RUN_MODE must be 'remote' or 'local'")
        return _model
