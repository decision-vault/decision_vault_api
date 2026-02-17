import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.schemas.hf_inference import HFInferenceRequest, HFInferenceResponse
from app.services.hf_model_loader import LocalModel, RemoteModel, get_model

router = APIRouter(prefix="/api/models", tags=["hf-inference"])
logger = logging.getLogger("decisionvault.hf")


def _truncate_prompt(tokenizer, prompt: str) -> tuple[str, int]:
    if not tokenizer:
        return prompt[: settings.hf_max_input_chars], min(len(prompt), settings.hf_max_input_chars)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > settings.hf_max_input_tokens:
        tokens = tokens[-settings.hf_max_input_tokens :]
    truncated = tokenizer.decode(tokens)
    return truncated, len(tokens)


@router.post("/hf-infer", response_model=HFInferenceResponse)
async def hf_infer(request: Request, payload: HFInferenceRequest):
    if settings.model_run_mode not in {"remote", "local"}:
        raise HTTPException(status_code=500, detail="MODEL_RUN_MODE misconfigured")

    model = await get_model()
    prompt, input_tokens = _truncate_prompt(getattr(model, "tokenizer", None), payload.prompt)

    logger.info(
        "hf_request",
        extra={
            "mode": settings.model_run_mode,
            "model": getattr(model, "model_name", "unknown"),
            "input_tokens": input_tokens,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
        },
    )

    try:
        if isinstance(model, RemoteModel):
            if model.mode == "lmstudio":
                url = f"{settings.lmstudio_base_url}{settings.lmstudio_chat_path}"
                headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
                body = {
                    "model": model.model_name,
                    "input": prompt,
                    "temperature": payload.temperature,
                }
                if payload.integrations:
                    body["integrations"] = payload.integrations
                if payload.context_length:
                    body["context_length"] = payload.context_length
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    if resp.status_code >= 400:
                        detail = "LM Studio request failed"
                        try:
                            payload = resp.json()
                            detail = payload.get("error", {}).get("message") or resp.text
                        except Exception:
                            detail = resp.text
                        logger.error(
                            "lmstudio_error",
                            extra={"status": resp.status_code, "body": resp.text[:500], "url": url},
                        )
                        raise HTTPException(status_code=400, detail=detail)
                    data = resp.json()
                output = data.get("output") or data.get("text") or ""
                if isinstance(output, list) and output:
                    first = output[0]
                    if isinstance(first, dict):
                        output = first.get("content") or first.get("text") or str(first)
                if not isinstance(output, str):
                    output = str(output)
                stats = data.get("stats") or {}
            else:
                completion = model.client.chat.completions.create(
                    model=model.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                )
                output = completion.choices[0].message.content or ""
            if model.mode == "lmstudio" and stats:
                total_tokens = int(stats.get("input_tokens", input_tokens)) + int(
                    stats.get("total_output_tokens", payload.max_tokens)
                )
            else:
                total_tokens = input_tokens + payload.max_tokens
            response = HFInferenceResponse(
                output=output,
                tokens=total_tokens,
                model=model.model_name,
            )
        elif isinstance(model, LocalModel):
            if hasattr(model.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": settings.hf_system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text = model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = model.tokenizer([text], return_tensors="pt")
            else:
                model_inputs = model.tokenizer([prompt], return_tensors="pt")
            if settings.local_device:
                model_inputs = {k: v.to(settings.local_device) for k, v in model_inputs.items()}
            generated = model.model.generate(
                model_inputs["input_ids"],
                max_new_tokens=payload.max_tokens,
                temperature=payload.temperature,
                do_sample=payload.temperature > 0.0,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs["input_ids"], generated)
            ]
            decoded = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = HFInferenceResponse(
                output=decoded,
                tokens=generated[0].numel(),
                model=model.model_name,
            )
        else:
            raise RuntimeError("Unknown model type")
    except Exception as exc:
        logger.exception("hf_infer_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Model inference failed")

    logger.info(
        "hf_response",
        extra={
            "mode": settings.model_run_mode,
            "model": response.model,
            "tokens": response.tokens,
            "output_preview": response.output[:200],
        },
    )

    # DecisionVault RAG context: this endpoint is for controlled generation
    # when augmenting decision evidence, never as a source of truth.
    return response
