from typing import Union
import logging
import httpx
import asyncio
from datetime import datetime, timezone
from bson import ObjectId

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.middleware.guard import withGuard
from app.schemas.prd_generation import (
    PRDClarificationResponse,
    PRDClarificationRespondRequest,
    PRDGenerateRequest,
    PRDGenerateResponse,
    PRDMultiStepResponse,
)
from app.services.prd_multistep_service import generate_multistep_prd
from app.services.prd_pg_service import (
    get_latest_prd_version,
    get_prd_version,
    list_prd_versions,
    store_prd_version,
)
from app.core.config import settings
from app.services.token_limiter import TokenBudget, TokenLimiter
from langchain_openai import ChatOpenAI
from app.db.mongo import get_db


router = APIRouter(prefix="/api/prd", tags=["prd"])
logger = logging.getLogger("decisionvault.prd.api")


PRIMARY_ORDER = ["project_name", "problem_statement", "target_users", "desired_features"]
TOTAL_REQUIRED_FIELDS = 4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_oid(value: str, field_name: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}")


def _as_aware_dt(value) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


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


async def _append_run_event(run_id: ObjectId, event: dict) -> None:
    db = get_db()
    now = _utcnow()
    stage = event.get("stage")
    status = event.get("status")
    if not stage:
        return
    if status == "running":
        # Set run started_at only once (first running stage).
        await db.prd_runs.update_one(
            {"_id": run_id, "started_at": None},
            {"$set": {"started_at": now}},
        )
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "running",
                    "updated_at": now,
                },
                "$push": {
                    "events": {"at": now, **event},
                    "steps": {
                        "stage": stage,
                        "status": "running",
                        "started_at": now,
                    },
                },
            },
        )
        return

    if status in {"completed", "failed"}:
        await db.prd_runs.update_one(
            {"_id": run_id, "steps.stage": stage},
            {
                "$set": {
                    "updated_at": now,
                    "steps.$.status": status,
                    "steps.$.ended_at": now,
                    "steps.$.input_tokens": event.get("input_tokens"),
                    "steps.$.output_tokens": event.get("output_tokens"),
                    "steps.$.retry_count": event.get("retry_count", 0),
                    "steps.$.error": event.get("error"),
                },
                "$push": {"events": {"at": now, **event}},
            },
        )
        return

    await db.prd_runs.update_one(
        {"_id": run_id},
        {"$set": {"updated_at": now}, "$push": {"events": {"at": now, **event}}},
    )


async def _run_prd_job(
    run_id: ObjectId,
    payload: PRDGenerateRequest,
    project_id: str,
    tenant_id: str,
    created_by: str,
) -> None:
    db = get_db()
    try:
        result = await generate_multistep_prd(
            payload,
            tenant_id=tenant_id,
            progress_cb=lambda ev: _append_run_event(run_id, ev),
        )
        stored = await store_prd_version(
            project_id=project_id,
            created_by=created_by,
            markdown_content=result.prd_markdown,
        )
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "result": {
                        "pages_estimated": result.pages_estimated,
                        "sections_generated": result.sections_generated,
                        "required_sections": result.required_sections,
                        "missing_sections": result.missing_sections,
                        "has_all_required_sections": result.has_all_required_sections,
                        "total_tokens_used": result.total_tokens_used,
                        "prd_markdown": result.prd_markdown,
                        "version": stored.get("version_number"),
                    },
                }
            },
        )
    except Exception as exc:
        logger.exception("prd_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "error": str(exc),
                }
            },
        )


async def _enqueue_prd_run(
    *,
    payload: PRDGenerateRequest,
    project_id: str,
    tenant_id: str,
    created_by: str,
) -> dict:
    db = get_db()
    run_id = ObjectId()
    now = _utcnow()
    run_doc = {
        "_id": run_id,
        "tenant_id": _as_oid(tenant_id, "tenant_id"),
        "project_id": _as_oid(project_id, "project_id"),
        "created_by": created_by,
        "status": "queued",
        "request": payload.model_dump(),
        "steps": [],
        "events": [{"at": now, "status": "queued"}],
        "error": None,
        "result": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    await db.prd_runs.insert_one(run_doc)
    asyncio.create_task(
        _run_prd_job(
            run_id=run_id,
            payload=payload,
            project_id=project_id,
            tenant_id=tenant_id,
            created_by=created_by,
        )
    )
    return {"run_id": str(run_id), "status": "queued"}


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
        multi = await generate_multistep_prd(payload, tenant_id=user.get("tenant_id"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("prd_generation_failed", extra={"error": str(exc), "project_id": project_id})
        raise HTTPException(status_code=500, detail=f"PRD generation failed: {str(exc)}")

    try:
        await store_prd_version(
            project_id=project_id,
            created_by=user.get("user_id"),
            markdown_content=multi.prd_markdown,
        )
    except RuntimeError as exc:
        # Postgres is optional for local/dev. PRD generation should still succeed.
        logger.warning("prd_store_skipped_runtime", extra={"error": str(exc), "project_id": project_id})
    except Exception as exc:
        # Do not fail generation because of persistence-only issues.
        logger.exception("prd_store_failed", extra={"error": str(exc), "project_id": project_id})

    return PRDGenerateResponse(
        status="ready_for_prd_generation",
        prd_markdown=multi.prd_markdown,
        confidence_score=0.95,
        sections_generated=multi.sections_generated,
    )


@router.post("/generate-multistep", response_model=PRDMultiStepResponse)
async def generate_prd_multistep(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    try:
        result = await generate_multistep_prd(payload, tenant_id=user.get("tenant_id"))
    except ValueError as exc:
        logger.warning(
            "prd_multistep_bad_request project_id=%s tenant_id=%s detail=%s",
            project_id,
            user.get("tenant_id"),
            str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("prd_multistep_failed", extra={"error": str(exc), "project_id": project_id})
        raise HTTPException(status_code=500, detail=f"Multi-step PRD generation failed: {str(exc)}")

    try:
        await store_prd_version(
            project_id=project_id,
            created_by=user.get("user_id"),
            markdown_content=result.prd_markdown,
        )
    except Exception:
        logger.warning("prd_multistep_store_skipped")
    return result


@router.post("/generate-multistep/run")
async def generate_prd_multistep_run(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        logger.warning(
            "prd_multistep_run_clarification_required project_id=%s tenant_id=%s completion_score=%s",
            project_id,
            user.get("tenant_id"),
            clarification.completion_score,
        )
        return {
            "status": clarification.status,
            "questions": clarification.questions,
            "completion_score": clarification.completion_score,
        }
    return await _enqueue_prd_run(
        payload=payload,
        project_id=project_id,
        tenant_id=user.get("tenant_id"),
        created_by=user.get("user_id"),
    )


@router.get("/runs/{run_id}")
async def get_prd_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    doc = await db.prd_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")

    now = _utcnow()
    steps = doc.get("steps", [])
    step_timings: list[dict] = []
    for step in steps:
        started_at = _as_aware_dt(step.get("started_at"))
        ended_at = _as_aware_dt(step.get("ended_at"))
        duration_seconds: float | None = None
        if started_at and ended_at:
            duration_seconds = round((ended_at - started_at).total_seconds(), 3)
        elif started_at and step.get("status") == "running":
            duration_seconds = round((now - started_at).total_seconds(), 3)
        step_timings.append(
            {
                **step,
                "duration_seconds": duration_seconds,
            }
        )

    run_started_at = _as_aware_dt(doc.get("started_at")) or _as_aware_dt(doc.get("created_at"))
    run_completed_at = _as_aware_dt(doc.get("completed_at"))
    total_elapsed_seconds: float | None = None
    if run_started_at and run_completed_at:
        total_elapsed_seconds = round((run_completed_at - run_started_at).total_seconds(), 3)
    elif run_started_at and doc.get("status") in {"queued", "running"}:
        total_elapsed_seconds = round((now - run_started_at).total_seconds(), 3)

    return {
        "run_id": str(doc["_id"]),
        "status": doc.get("status"),
        "steps": step_timings,
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {
            "total_elapsed_seconds": total_elapsed_seconds,
        },
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/clarification/respond")
async def respond_prd_clarification(
    payload: PRDClarificationRespondRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    merged = _apply_clarification_answers(payload.draft, payload.answers)
    clarification = _build_clarification(merged)
    if clarification and clarification.questions:
        return {
            "status": clarification.status,
            "questions": clarification.questions,
            "completion_score": clarification.completion_score,
        }
    return await _enqueue_prd_run(
        payload=merged,
        project_id=project_id,
        tenant_id=user.get("tenant_id"),
        created_by=user.get("user_id"),
    )


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


@router.get("/latest")
async def get_latest_prd(
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    latest = await get_latest_prd_version(project_id)
    if not latest:
        raise HTTPException(status_code=404, detail="PRD not found")
    return {
        "project_id": latest["project_id"],
        "version": latest["version_number"],
        "created_by": latest["created_by"],
        "created_at": latest["created_at"],
        "content": latest["markdown_content"],
        "source": "llm",
    }


@router.get("/versions")
async def get_prd_versions(
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    versions = await list_prd_versions(project_id)
    return {"items": versions}


@router.get("/versions/{version_number}")
async def get_prd_by_version(
    version_number: int,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    doc = await get_prd_version(project_id, version_number)
    if not doc:
        raise HTTPException(status_code=404, detail="PRD version not found")
    return {
        "project_id": doc["project_id"],
        "version": doc["version_number"],
        "created_by": doc["created_by"],
        "created_at": doc["created_at"],
        "content": doc["markdown_content"],
        "source": "llm",
    }
