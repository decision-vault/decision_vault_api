from typing import Union
import logging
import httpx

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.middleware.guard import withGuard
from app.schemas.prd_generation import (
    PRDClarificationResponse,
    PRDClarificationRespondRequest,
    PRDGenerateRequest,
    PRDGenerateResponse,
)
from app.services.prd_graph_service import run_prd_pipeline
from app.services.prd_pg_service import store_prd_version
from app.core.config import settings
from app.services.token_limiter import TokenBudget, TokenLimiter
from langchain_openai import ChatOpenAI


router = APIRouter(prefix="/api/prd", tags=["prd"])
logger = logging.getLogger("decisionvault.prd.api")


PRIMARY_ORDER = ["project_name", "problem_statement", "target_users", "desired_features"]
TOTAL_REQUIRED_FIELDS = 4


def _resolve_llm_stream_config() -> tuple[str, str, str | None]:
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "huggingface":
        model = settings.hf_openai_model or settings.llm_model
        api_key = settings.hf_api_token
        base_url = settings.hf_router_base_url
    elif provider == "lmstudio":
        model = settings.lmstudio_model or settings.llm_model
        api_key = settings.llm_api_key or "lm-studio"
        base_url = settings.lmstudio_base_url
    else:
        model = settings.llm_model or settings.llm_strong_model
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
    logger.warning(
        "prd_stream_llm_selected provider=%s model=%s base_url=%s",
        provider or "default",
        model,
        base_url,
    )
    return model, api_key, base_url


def _extract_lmstudio_text(payload: dict) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                chunks.append(item["content"].strip())
        if chunks:
            return "\n".join(chunks).strip()
    return ""


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        return len([item for item in value if isinstance(item, str) and item.strip()]) > 0
    return bool(value)


def _build_clarification(payload: PRDGenerateRequest) -> PRDClarificationResponse | None:
    missing_primary: list[str] = []
    if not _is_filled(payload.title):
        missing_primary.append("project_name")
    if not _is_filled(payload.problem_statement):
        missing_primary.append("problem_statement")
    if not _is_filled(payload.target_users):
        missing_primary.append("target_users")
    cleaned_features = [item.strip() for item in payload.features if item and item.strip()]
    if len(cleaned_features) < 3:
        missing_primary.append("desired_features")

    filled_primary = TOTAL_REQUIRED_FIELDS - len(missing_primary)
    completion_score = round(filled_primary / TOTAL_REQUIRED_FIELDS, 2)

    if missing_primary or completion_score < 0.6:
        questions: list[str] = []
        for field in PRIMARY_ORDER:
            if field == "project_name" and field in missing_primary:
                questions.append("What is the project name?")
            elif field == "problem_statement" and field in missing_primary:
                questions.append("Please provide a clear problem statement (minimum 100 characters).")
            elif field == "target_users" and field in missing_primary:
                questions.append("Who are the target users for this product?")
            elif field == "desired_features" and field in missing_primary:
                questions.append("List at least 3 core desired features.")
        return PRDClarificationResponse(
            status="clarification_required",
            questions=questions[:5],
            completion_score=completion_score,
        )

    # Primary required fields are complete; proceed to PRD generation.
    return None


def _split_feature_text(value: str) -> list[str]:
    tokens = [item.strip() for item in value.replace("\n", ",").split(",")]
    return [item for item in tokens if item]


def _apply_clarification_answers(
    draft: PRDGenerateRequest,
    answers: dict[str, str | list[str]],
) -> PRDGenerateRequest:
    title = draft.title
    problem_statement = draft.problem_statement
    target_users = draft.target_users
    features = list(draft.features)
    additional_notes = draft.additional_notes

    for key, value in answers.items():
        lowered = key.lower()
        if isinstance(value, list):
            value_text = ", ".join([str(v).strip() for v in value if str(v).strip()])
        else:
            value_text = str(value).strip()

        if not value_text:
            continue

        if "project" in lowered and "name" in lowered:
            title = value_text
        elif "problem" in lowered:
            problem_statement = value_text
        elif "target" in lowered and "user" in lowered:
            target_users = value_text
        elif "feature" in lowered:
            parsed = value if isinstance(value, list) else _split_feature_text(value_text)
            features = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            # Keep unknown answers in notes so user context is preserved.
            additional_notes = f"{additional_notes or ''}\n{key}: {value_text}".strip()

    return PRDGenerateRequest(
        title=title,
        problem_statement=problem_statement,
        target_users=target_users,
        features=features,
        additional_notes=additional_notes,
    )


@router.post("/generate", response_model=Union[PRDGenerateResponse, PRDClarificationResponse])
async def generate_prd(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})

    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        return clarification

    try:
        result = await run_prd_pipeline(
            {
                **payload.model_dump(),
                "tenant_id": user.get("tenant_id"),
            }
        )
    except TimeoutError:
        raise HTTPException(status_code=500, detail="PRD generation timed out")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("prd_generation_failed", extra={"error": str(exc), "project_id": project_id})
        raise HTTPException(status_code=500, detail=f"PRD generation failed: {str(exc)}")

    try:
        await store_prd_version(
            project_id=project_id,
            created_by=user.get("user_id"),
            markdown_content=result["prd_markdown"],
        )
    except RuntimeError as exc:
        # Postgres is optional for local/dev. PRD generation should still succeed.
        logger.warning("prd_store_skipped_runtime", extra={"error": str(exc), "project_id": project_id})
    except Exception as exc:
        # Do not fail generation because of persistence-only issues.
        logger.exception("prd_store_failed", extra={"error": str(exc), "project_id": project_id})

    return PRDGenerateResponse(
        status="ready_for_prd_generation",
        prd_markdown=result["prd_markdown"],
        confidence_score=result["confidence_score"],
        sections_generated=result["sections_generated"],
    )


@router.post("/clarification/respond", response_model=Union[PRDGenerateResponse, PRDClarificationResponse])
async def respond_prd_clarification(
    payload: PRDClarificationRespondRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    merged = _apply_clarification_answers(payload.draft, payload.answers)
    return await generate_prd(merged, project_id, user)


@router.post("/generate/stream")
async def generate_prd_stream(
    request: Request,
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        raise HTTPException(
            status_code=400,
            detail={
                "status": clarification.status,
                "questions": clarification.questions,
                "completion_score": clarification.completion_score,
            },
        )

    limiter = TokenLimiter(TokenBudget(settings.llm_max_input_tokens, settings.llm_max_output_tokens))
    input_tokens = max(1, int(len(payload.model_dump_json().split()) * 1.3))
    limiter.enforce(input_tokens=input_tokens, output_tokens=settings.llm_max_output_tokens)

    model, api_key, base_url = _resolve_llm_stream_config()
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    provider = (settings.llm_provider or "").strip().lower()
    prompt = (
        f"Title: {payload.title}\n"
        f"Problem: {payload.problem_statement}\n"
        f"Target users: {payload.target_users}\n"
        f"Features: {', '.join(payload.features)}\n"
        f"Additional notes: {payload.additional_notes or 'None'}\n\n"
        "Generate PRD markdown. No invented data."
    )

    async def event_stream():
        if provider == "lmstudio":
            url = f"{(settings.lmstudio_base_url or '').rstrip('/')}{settings.lmstudio_chat_path}"
            body = {
                "model": settings.lmstudio_model or model,
                "input": prompt,
                "temperature": 0.2,
            }
            logger.warning(
                "prd_stream_lmstudio_request base_url=%s chat_path=%s model=%s",
                settings.lmstudio_base_url,
                settings.lmstudio_chat_path,
                body["model"],
            )
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                data = resp.json()
            text = _extract_lmstudio_text(data)
            if text:
                logger.warning(
                    "prd_stream_lmstudio_response model=%s output_chars=%s",
                    body["model"],
                    len(text),
                )
                yield text
            return

        llm = ChatOpenAI(
            model=model,
            temperature=0.2,
            top_p=0.8,
            max_tokens=settings.llm_max_output_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        async for chunk in llm.astream(prompt):
            if await request.is_disconnected():
                break
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text

    return StreamingResponse(event_stream(), media_type="text/markdown")
