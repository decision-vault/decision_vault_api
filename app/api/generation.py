from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_db
from app.middleware.guard import withGuard

# We reuse existing run endpoints as internal helpers to avoid duplicating orchestration logic.
from app.api.prd import (
    generate_prd_multistep_run as _start_prd_run,
    get_prd_run_status as _get_prd_status,
    pause_prd_run as _pause_prd,
    resume_prd_run as _resume_prd,
    stop_prd_run as _stop_prd,
    respond_prd_run_clarification as _respond_prd_clarification,
    retry_prd_run as _retry_prd,
)
from app.api.requirements import (
    generate_system_design_run as _start_sdd_run,
    get_system_design_run_status as _get_sdd_status,
    pause_system_design_run as _pause_sdd,
    resume_system_design_run as _resume_sdd,
    stop_system_design_run as _stop_sdd,
    generate_schema_flow_run as _start_schema_run,
    get_schema_flow_run_status as _get_schema_status,
    pause_schema_flow_run as _pause_schema,
    resume_schema_flow_run as _resume_schema,
    stop_schema_flow_run as _stop_schema,
    generate_usecase_flow_run as _start_usecase_run,
    get_usecase_flow_run_status as _get_usecase_status,
    pause_usecase_flow_run as _pause_usecase,
    resume_usecase_flow_run as _resume_usecase,
    stop_usecase_flow_run as _stop_usecase,
    generate_sequence_flow_run as _start_sequence_run,
    get_sequence_flow_run_status as _get_sequence_status,
    pause_sequence_flow_run as _pause_sequence,
    resume_sequence_flow_run as _resume_sequence,
    stop_sequence_flow_run as _stop_sequence,
    generate_architecture_diagram_doc as _generate_architecture,
    _serialize_chat_messages as _serialize_chat_messages,
    _trim_chat_messages_one_by_one as _trim_chat_messages_one_by_one,
    _dedupe_questions_by_field_key as _dedupe_questions_by_field_key,
)
from app.schemas.prd_generation import PRDGenerateRequest


router = APIRouter(prefix="/api/generation", tags=["generation"])

GenerationKind = Literal["prd", "sdd", "schema", "usecase", "sequence", "architecture"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_oid(value: str | None, field: str) -> ObjectId:
    if not value:
        raise HTTPException(status_code=400, detail={field: f"{field} is required"})
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail={field: f"Invalid {field}"})


async def _maybe_attach_doc_id(
    *,
    kind: GenerationKind,
    tenant_id: str,
    project_id: str,
    intake_id: str | None,
    status_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    For non-PRD runs (and older PRD shapes), try to attach a doc_id for the latest version
    so the UI can open via /api/docs/{doc_id}.
    """
    try:
        result = status_payload.get("result") or {}
        if not isinstance(result, dict):
            return status_payload
        if kind == "prd":
            return status_payload
        version = result.get("version")
        if version is None:
            return status_payload
        db = get_db()
        tenant_oid = ObjectId(tenant_id)
        project_oid = ObjectId(project_id)
        intake_oid = ObjectId(intake_id) if intake_id else None

        doc_id = None
        if kind == "sdd":
            if not intake_oid:
                return status_payload
            doc = await db.system_design_documents.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "intake_id": intake_oid,
                    "version": int(version),
                },
                {"_id": 1},
            )
            doc_id = str(doc.get("_id")) if doc else None
        elif kind == "schema":
            if not intake_oid:
                return status_payload
            doc = await db.schema_flow_documents.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "intake_id": intake_oid,
                    "version": int(version),
                },
                {"_id": 1},
            )
            doc_id = str(doc.get("_id")) if doc else None
        elif kind == "usecase":
            if not intake_oid:
                return status_payload
            doc = await db.usecase_flow_documents.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "intake_id": intake_oid,
                    "version": int(version),
                },
                {"_id": 1},
            )
            doc_id = str(doc.get("_id")) if doc else None
        elif kind == "sequence":
            if not intake_oid:
                return status_payload
            doc = await db.sequence_flow_documents.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "intake_id": intake_oid,
                    "version": int(version),
                },
                {"_id": 1},
            )
            doc_id = str(doc.get("_id")) if doc else None
        elif kind == "architecture":
            if not intake_oid:
                return status_payload
            doc = await db.architecture_diagram_documents.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "intake_id": intake_oid,
                    "version": int(version),
                },
                {"_id": 1},
            )
            doc_id = str(doc.get("_id")) if doc else None

        if doc_id:
            result["doc_id"] = doc_id
            status_payload["result"] = result
        return status_payload
    except Exception:
        return status_payload


@router.post("/runs")
async def start_generation_run(
    body: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    kind: GenerationKind = str(body.get("kind") or "").strip().lower()  # type: ignore[assignment]
    if kind not in {"prd", "sdd", "schema", "usecase", "sequence", "architecture"}:
        raise HTTPException(status_code=400, detail={"kind": "Unsupported kind"})
    intake_id = str(body.get("intake_id") or "").strip()
    payload = body.get("payload") or {}
    replace_active = bool(body.get("replace_active") or False)
    if kind != "prd" and not intake_id:
        raise HTTPException(status_code=400, detail={"intake_id": "intake_id is required"})

    db = get_db()
    gen_run_id = ObjectId()
    now = _utcnow()
    tenant_oid = _as_oid(str(user.get("tenant_id") or ""), "tenant_id")
    project_oid = _as_oid(project_id, "project_id")
    intake_oid = _as_oid(intake_id, "intake_id") if intake_id else None

    await db.generation_runs.insert_one(
        {
            "_id": gen_run_id,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
            "intake_id": intake_oid,
            "kind": kind,
            "status": "queued",
            "child_run_id": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )

    async def _stop_child(kind_value: GenerationKind, child_id: str) -> None:
        if not child_id:
            return
        if kind_value == "prd":
            await _stop_prd(child_id, project_id=project_id, user=user)
        elif kind_value == "sdd":
            await _stop_sdd(child_id, project_id=project_id, user=user)
        elif kind_value == "schema":
            await _stop_schema(child_id, project_id=project_id, user=user)
        elif kind_value == "usecase":
            await _stop_usecase(child_id, project_id=project_id, user=user)
        elif kind_value == "sequence":
            await _stop_sequence(child_id, project_id=project_id, user=user)
        else:
            return

    # Regenerate path: stop any active run of the same kind for this project, then start a new one.
    if replace_active:
        try:
            active = await db.generation_runs.find_one(
                {
                    "tenant_id": tenant_oid,
                    "project_id": project_oid,
                    "kind": kind,
                    "status": {"$in": ["queued", "running", "paused", "clarification_required"]},
                },
                sort=[("created_at", -1)],
            )
            if active:
                prev_child = str(active.get("child_run_id") or "").strip()
                # Best-effort stop; even if it fails, we still proceed to start a new run.
                try:
                    await _stop_child(kind, prev_child)
                except Exception:
                    pass
                await db.generation_runs.update_one(
                    {"_id": active["_id"]},
                    {"$set": {"status": "stopped", "updated_at": _utcnow(), "completed_at": _utcnow(), "error": "Superseded by regenerate."}},
                )
        except Exception:
            pass

    # Start underlying run.
    child_run_id = None
    started_status = "queued"
    try:
        if kind == "prd":
            # Thread intake_id through to PRD runs so the completion can append to intake chat_messages.
            if intake_id and isinstance(payload, dict):
                payload = {**payload, "intake_id": intake_id}
            prd_req = PRDGenerateRequest.model_validate(payload)
            resp = await _start_prd_run(prd_req, project_id=project_id, user=user)
            # Clarification path passes through.
            if isinstance(resp, dict) and resp.get("status") == "clarification_required":
                await db.generation_runs.update_one(
                    {"_id": gen_run_id},
                    {"$set": {"status": "clarification_required", "updated_at": _utcnow()}},
                )
                return {"run_id": str(gen_run_id), "kind": kind, "status": "clarification_required", **resp}
            child_run_id = str((resp or {}).get("run_id") or "")
            started_status = str((resp or {}).get("status") or "queued")
        elif kind == "sdd":
            resp = await _start_sdd_run(intake_id, project_id=project_id, user=user)
            child_run_id = str((resp or {}).get("run_id") or "")
            started_status = str((resp or {}).get("status") or "queued")
        elif kind == "schema":
            resp = await _start_schema_run(intake_id, payload if isinstance(payload, dict) else {}, project_id=project_id, user=user)
            child_run_id = str((resp or {}).get("run_id") or "")
            started_status = str((resp or {}).get("status") or "queued")
        elif kind == "usecase":
            resp = await _start_usecase_run(intake_id, payload if isinstance(payload, dict) else {}, project_id=project_id, user=user)
            child_run_id = str((resp or {}).get("run_id") or "")
            started_status = str((resp or {}).get("status") or "queued")
        elif kind == "sequence":
            resp = await _start_sequence_run(intake_id, payload if isinstance(payload, dict) else {}, project_id=project_id, user=user)
            child_run_id = str((resp or {}).get("run_id") or "")
            started_status = str((resp or {}).get("status") or "queued")
        else:
            # Architecture is synchronous today; treat it as an instant run.
            resp = await _generate_architecture(intake_id, payload if isinstance(payload, dict) else {}, project_id=project_id, user=user)
            started_status = "completed"
            await db.generation_runs.update_one(
                {"_id": gen_run_id},
                {"$set": {"status": "completed", "updated_at": _utcnow(), "completed_at": _utcnow()}},
            )
            return {"run_id": str(gen_run_id), "kind": kind, "status": "completed", "result": resp}

        if not child_run_id:
            raise ValueError("Failed to start child run")
        await db.generation_runs.update_one(
            {"_id": gen_run_id},
            {"$set": {"child_run_id": child_run_id, "status": started_status or "queued", "updated_at": _utcnow()}},
        )
        return {"run_id": str(gen_run_id), "kind": kind, "status": started_status or "queued", "child_run_id": child_run_id}
    except Exception as exc:
        await db.generation_runs.update_one(
            {"_id": gen_run_id},
            {"$set": {"status": "failed", "error": str(exc), "updated_at": _utcnow(), "completed_at": _utcnow()}},
        )
        raise


@router.get("/runs/{run_id}")
async def get_generation_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()

    if kind == "architecture" and not child_run_id:
        return {"run_id": run_id, "kind": kind, "status": gen.get("status"), "error": gen.get("error"), "result": gen.get("result")}

    if not child_run_id:
        return {"run_id": run_id, "kind": kind, "status": gen.get("status"), "error": gen.get("error")}

    intake_id = str(gen.get("intake_id") or "")
    status_payload: dict[str, Any]
    if kind == "prd":
        status_payload = await _get_prd_status(child_run_id, project_id=project_id, user=user)
    elif kind == "sdd":
        status_payload = await _get_sdd_status(child_run_id, project_id=project_id, user=user)
    elif kind == "schema":
        status_payload = await _get_schema_status(child_run_id, project_id=project_id, user=user)
    elif kind == "usecase":
        status_payload = await _get_usecase_status(child_run_id, project_id=project_id, user=user)
    else:
        status_payload = await _get_sequence_status(child_run_id, project_id=project_id, user=user)

    unified_status = str(status_payload.get("status") or gen.get("status") or "")
    await db.generation_runs.update_one(
        {"_id": gen["_id"]},
        {"$set": {"status": unified_status, "updated_at": _utcnow(), "error": status_payload.get("error")}},
    )
    if unified_status in {"completed", "failed", "stopped"} and not gen.get("completed_at"):
        await db.generation_runs.update_one({"_id": gen["_id"]}, {"$set": {"completed_at": _utcnow()}})

    status_payload = await _maybe_attach_doc_id(
        kind=kind,
        tenant_id=str(user.get("tenant_id") or ""),
        project_id=project_id,
        intake_id=str(gen.get("intake_id") or "") or None,
        status_payload=status_payload,
    )

    chat_messages = None
    try:
        intake_oid = gen.get("intake_id")
        if intake_oid:
            intake_doc = await db.requirements_intakes.find_one(
                {"_id": intake_oid, "tenant_id": gen.get("tenant_id"), "project_id": gen.get("project_id")},
                {"chat_messages": 1},
            )
            if intake_doc and isinstance(intake_doc.get("chat_messages"), list):
                trimmed = _dedupe_questions_by_field_key(_trim_chat_messages_one_by_one(intake_doc.get("chat_messages") or []))
                chat_messages = _serialize_chat_messages(trimmed)
    except Exception:
        chat_messages = None
    return {
        "run_id": str(gen.get("_id")),
        "kind": kind,
        "status": unified_status,
        "child_run_id": child_run_id,
        "error": status_payload.get("error"),
        "result": status_payload.get("result"),
        "partial_result": status_payload.get("partial_result"),
        "clarification": status_payload.get("clarification"),
        "steps": status_payload.get("steps") or [],
        "timing": status_payload.get("timing") or {},
        **({"chat_messages": chat_messages} if chat_messages is not None else {}),
        "created_at": gen.get("created_at"),
        "updated_at": gen.get("updated_at"),
        "started_at": gen.get("started_at"),
        "completed_at": gen.get("completed_at"),
    }


@router.get("/active")
async def get_active_generation_runs(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    cursor = db.generation_runs.find(
        {
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
            "status": {"$in": ["queued", "running", "paused", "clarification_required"]},
        }
    ).sort("created_at", -1)
    items = []
    async for doc in cursor:
        items.append(
            {
                "run_id": str(doc.get("_id")),
                "kind": doc.get("kind"),
                "status": doc.get("status"),
                "child_run_id": doc.get("child_run_id"),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
            }
        )
    return {"items": items}


@router.post("/runs/{run_id}/pause")
async def pause_generation_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()
    if not child_run_id:
        return {"run_id": run_id, "status": gen.get("status")}
    if kind == "prd":
        resp = await _pause_prd(child_run_id, project_id=project_id, user=user)
    elif kind == "sdd":
        resp = await _pause_sdd(child_run_id, project_id=project_id, user=user)
    elif kind == "schema":
        resp = await _pause_schema(child_run_id, project_id=project_id, user=user)
    elif kind == "usecase":
        resp = await _pause_usecase(child_run_id, project_id=project_id, user=user)
    else:
        resp = await _pause_sequence(child_run_id, project_id=project_id, user=user)
    await db.generation_runs.update_one({"_id": gen["_id"]}, {"$set": {"status": resp.get("status"), "updated_at": _utcnow()}})
    return {"run_id": run_id, "kind": kind, "status": resp.get("status")}


@router.post("/runs/{run_id}/resume")
async def resume_generation_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()
    if not child_run_id:
        return {"run_id": run_id, "status": gen.get("status")}
    if kind == "prd":
        resp = await _resume_prd(child_run_id, project_id=project_id, user=user)
    elif kind == "sdd":
        resp = await _resume_sdd(child_run_id, project_id=project_id, user=user)
    elif kind == "schema":
        resp = await _resume_schema(child_run_id, project_id=project_id, user=user)
    elif kind == "usecase":
        resp = await _resume_usecase(child_run_id, project_id=project_id, user=user)
    else:
        resp = await _resume_sequence(child_run_id, project_id=project_id, user=user)
    await db.generation_runs.update_one({"_id": gen["_id"]}, {"$set": {"status": resp.get("status"), "updated_at": _utcnow()}})
    return {"run_id": run_id, "kind": kind, "status": resp.get("status")}


@router.post("/runs/{run_id}/stop")
async def stop_generation_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()
    if not child_run_id:
        await db.generation_runs.update_one(
            {"_id": gen["_id"]},
            {"$set": {"status": "stopped", "updated_at": _utcnow(), "completed_at": _utcnow(), "error": "Run stopped by user."}},
        )
        return {"run_id": run_id, "kind": kind, "status": "stopped"}
    if kind == "prd":
        resp = await _stop_prd(child_run_id, project_id=project_id, user=user)
    elif kind == "sdd":
        resp = await _stop_sdd(child_run_id, project_id=project_id, user=user)
    elif kind == "schema":
        resp = await _stop_schema(child_run_id, project_id=project_id, user=user)
    elif kind == "usecase":
        resp = await _stop_usecase(child_run_id, project_id=project_id, user=user)
    else:
        resp = await _stop_sequence(child_run_id, project_id=project_id, user=user)
    await db.generation_runs.update_one(
        {"_id": gen["_id"]},
        {"$set": {"status": resp.get("status"), "updated_at": _utcnow(), "completed_at": _utcnow(), "error": "Run stopped by user."}},
    )
    return {"run_id": run_id, "kind": kind, "status": resp.get("status")}


@router.post("/runs/{run_id}/clarification/respond")
async def respond_generation_run_clarification(
    run_id: str,
    body: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    """
    Unified clarification responder. Currently supports PRD runs only.
    """
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()
    if kind != "prd" or not child_run_id:
        raise HTTPException(status_code=400, detail="Clarifications are only supported for PRD runs")
    return await _respond_prd_clarification(child_run_id, body, project_id=project_id, user=user)


@router.post("/runs/{run_id}/retry")
async def retry_generation_run(
    run_id: str,
    body: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    """
    Retry a failed/stopped generation run without creating a new run_id.
    Currently supported for PRD runs only.
    """
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    gen = await db.generation_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(str(user.get("tenant_id") or ""), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not gen:
        raise HTTPException(status_code=404, detail="Generation run not found")
    kind: GenerationKind = str(gen.get("kind") or "").strip().lower()  # type: ignore[assignment]
    child_run_id = str(gen.get("child_run_id") or "").strip()
    if kind != "prd" or not child_run_id:
        raise HTTPException(status_code=400, detail="Retry is only supported for PRD runs")

    resp = await _retry_prd(child_run_id, body, project_id=project_id, user=user)
    # Mirror status to the unified run doc.
    await db.generation_runs.update_one(
        {"_id": gen["_id"]},
        {"$set": {"status": str(resp.get("status") or "queued"), "updated_at": _utcnow(), "error": None, "completed_at": None, "started_at": None}},
    )
    return {"run_id": run_id, "kind": kind, "child_run_id": child_run_id, **resp}
