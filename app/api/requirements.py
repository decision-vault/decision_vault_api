import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from app.db.mongo import get_db

from app.middleware.guard import withGuard
from app.schemas.requirements import (
    RequirementsGenerateRequest,
    RequirementsGenerateResponse,
    RequirementsRespondRequest,
    RequirementsRespondResponse,
    RequirementsStartRequest,
    RequirementsStartResponse,
)
from app.services.requirements_service import generate_prd, respond_intake, start_intake, undo_intake, redo_intake
from app.services.prd_service import generate_prd as generate_prd_text, save_prd
from app.services.schema_flow_service import generate_schema_flow
from app.services.usecase_flow_service import generate_usecase_flow
from app.services.architecture_mermaid_service import generate_architecture_mermaid
from app.services.system_design_service import generate_system_design


router = APIRouter(prefix="/api/requirements", tags=["requirements"])
logger = logging.getLogger("decisionvault.requirements.sdd")
_ACTIVE_SDD_TASKS_BY_RUN_ID: dict[str, asyncio.Task] = {}
_ACTIVE_SCHEMA_TASKS_BY_RUN_ID: dict[str, asyncio.Task] = {}
_ACTIVE_USECASE_TASKS_BY_RUN_ID: dict[str, asyncio.Task] = {}
_ACTIVE_SEQUENCE_TASKS_BY_RUN_ID: dict[str, asyncio.Task] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_oid(value: str, field: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field}")


def _sequence_to_mermaid(nodes: list[dict], edges: list[dict]) -> str:
    participants: dict[str, str] = {}
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        name = str((node.get("data") or {}).get("name") or node_id).strip()
        participants[node_id] = name or node_id

    lines: list[str] = ["sequenceDiagram"]
    for node_id, name in participants.items():
        alias = node_id.replace("-", "_")
        lines.append(f"    participant {alias} as {name}")

    for edge in edges or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in participants or target not in participants:
            continue
        label = str(edge.get("label") or (edge.get("data") or {}).get("label") or "").strip() or "interaction"
        lines.append(f"    {source.replace('-', '_')}->>{target.replace('-', '_')}: {label}")

    return "\n".join(lines)


async def _save_system_design(intake_id: str, project_id: str, tenant_id: str, content: str) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    last = await db.system_design_documents.find_one(
        {"intake_id": intake_oid},
        sort=[("version", -1)],
    )
    next_version = (last.get("version") if last else 0) + 1
    doc = {
        "intake_id": intake_oid,
        "project_id": ObjectId(project_id),
        "tenant_id": ObjectId(tenant_id),
        "version": next_version,
        "generated_at": _utcnow(),
        "content": content,
    }
    await db.system_design_documents.insert_one(doc)
    return doc


async def _append_sdd_run_event(run_id: ObjectId, event: dict) -> None:
    db = get_db()
    stage = event.get("stage")
    status = event.get("status")
    now = _utcnow()
    if not stage:
        return
    existing = await db.sdd_runs.find_one({"_id": run_id, "steps.stage": stage}, {"_id": 1})
    if existing:
        update_doc = {
            "steps.$.status": status,
            "steps.$.updated_at": now,
        }
        if event.get("started_at"):
            update_doc["steps.$.started_at"] = event.get("started_at")
        if event.get("completed_at"):
            update_doc["steps.$.completed_at"] = event.get("completed_at")
        if event.get("duration_seconds") is not None:
            update_doc["steps.$.duration_seconds"] = event.get("duration_seconds")
        if event.get("input_tokens") is not None:
            update_doc["steps.$.input_tokens"] = event.get("input_tokens")
        if event.get("output_tokens") is not None:
            update_doc["steps.$.output_tokens"] = event.get("output_tokens")
        if event.get("retry_count") is not None:
            update_doc["steps.$.retry_count"] = event.get("retry_count")
        if event.get("error"):
            update_doc["steps.$.error"] = event.get("error")
        await db.sdd_runs.update_one({"_id": run_id, "steps.stage": stage}, {"$set": update_doc})
    else:
        await db.sdd_runs.update_one(
            {"_id": run_id},
            {
                "$push": {
                    "steps": {
                        "stage": stage,
                        "status": status or "queued",
                        "started_at": event.get("started_at"),
                        "completed_at": event.get("completed_at"),
                        "duration_seconds": event.get("duration_seconds"),
                        "input_tokens": event.get("input_tokens"),
                        "output_tokens": event.get("output_tokens"),
                        "retry_count": event.get("retry_count"),
                        "error": event.get("error"),
                        "updated_at": now,
                    }
                },
                "$set": {"updated_at": now},
            },
        )


async def _run_sdd_job(
    *,
    run_id: ObjectId,
    intake_id: str,
    project_id: str,
    tenant_id: str,
    structured: dict,
) -> None:
    db = get_db()
    started_at = _utcnow()
    await db.sdd_runs.update_one(
        {"_id": run_id},
        {"$set": {"status": "running", "started_at": started_at, "updated_at": started_at}},
    )
    try:
        async def _progress_handler(ev: dict) -> None:
            await _append_sdd_run_event(run_id, ev)
            while True:
                run_doc = await db.sdd_runs.find_one({"_id": run_id}, {"pause_requested": 1, "stop_requested": 1})
                if not run_doc:
                    return
                if run_doc.get("stop_requested"):
                    raise asyncio.CancelledError("Run stopped by user.")
                if run_doc.get("pause_requested"):
                    await db.sdd_runs.update_one(
                        {"_id": run_id},
                        {"$set": {"status": "paused", "updated_at": _utcnow()}},
                    )
                    await asyncio.sleep(1)
                    continue
                break
            await db.sdd_runs.update_one(
                {"_id": run_id},
                {"$set": {"status": "running", "updated_at": _utcnow()}},
            )

        content = await generate_system_design(
            structured,
            tenant_id,
            project_id=project_id,
            intake_id=intake_id,
            run_id=str(run_id),
            progress_cb=_progress_handler,
        )
        run_doc = await db.sdd_runs.find_one({"_id": run_id}, {"stop_requested": 1})
        if run_doc and run_doc.get("stop_requested"):
            raise asyncio.CancelledError("Run stopped by user.")
        saved = await _save_system_design(intake_id, project_id, tenant_id, content)
        completed_at = _utcnow()
        await db.sdd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "error": None,
                    "result": {
                        "version": saved.get("version"),
                        "generated_at": saved.get("generated_at"),
                    },
                }
            },
        )
    except asyncio.CancelledError:
        stopped_at = _utcnow()
        await db.sdd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "completed_at": stopped_at,
                    "updated_at": stopped_at,
                    "error": "Run stopped by user.",
                }
            },
        )
    except Exception as exc:
        logger.exception("sdd_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        failed_at = _utcnow()
        await db.sdd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": failed_at,
                    "updated_at": failed_at,
                    "error": str(exc),
                }
            },
        )
    finally:
        _ACTIVE_SDD_TASKS_BY_RUN_ID.pop(str(run_id), None)


async def _save_schema_flow(
    *,
    intake_id: str,
    tenant_id: str,
    project_id: str,
    result: dict,
) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    last = await db.schema_flow_documents.find_one(
        {
            "intake_id": intake_oid,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
        },
        sort=[("version", -1)],
    )
    next_version = (last.get("version") if last else 0) + 1
    now = _utcnow()
    doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": result.get("nodes") or [],
        "edges": result.get("edges") or [],
        "summary": result.get("summary") or "",
        "updated_at": now,
    }
    await db.schema_flow_documents.insert_one(doc)
    latest_doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": doc["nodes"],
        "edges": doc["edges"],
        "summary": doc["summary"],
        "updated_at": now,
    }
    await db.schema_flows.update_one(
        {
            "intake_id": intake_oid,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
        },
        {"$set": latest_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return doc


async def _get_latest_sdd_content(*, tenant_id: str, project_id: str) -> str:
    db = get_db()
    doc = await db.system_design_documents.find_one(
        {
            "tenant_id": ObjectId(tenant_id),
            "project_id": ObjectId(project_id),
        },
        sort=[("generated_at", -1)],
    )
    return str(doc.get("content") or "") if doc else ""


async def _save_usecase_flow(
    *,
    intake_id: str,
    tenant_id: str,
    project_id: str,
    result: dict,
) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    last = await db.usecase_flow_documents.find_one(
        {
            "intake_id": intake_oid,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
        },
        sort=[("version", -1)],
    )
    next_version = (last.get("version") if last else 0) + 1
    now = _utcnow()
    doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": result.get("nodes") or [],
        "edges": result.get("edges") or [],
        "summary": result.get("summary") or "",
        "updated_at": now,
    }
    await db.usecase_flow_documents.insert_one(doc)
    latest_doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": doc["nodes"],
        "edges": doc["edges"],
        "summary": doc["summary"],
        "updated_at": now,
    }
    await db.usecase_flows.update_one(
        {"intake_id": intake_oid, "tenant_id": tenant_oid, "project_id": project_oid},
        {"$set": latest_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return doc


async def _run_usecase_flow_job(
    *,
    run_id: ObjectId,
    intake_id: str,
    project_id: str,
    tenant_id: str,
    structured: dict,
    user_request: str,
    current_nodes: list,
    current_edges: list,
    latest_sdd_content: str,
) -> None:
    db = get_db()
    started_at = _utcnow()
    await db.usecase_flow_runs.update_one(
        {"_id": run_id},
        {
            "$set": {
                "status": "running",
                "started_at": started_at,
                "updated_at": started_at,
                "steps": [{"stage": "usecase_generation", "status": "running", "started_at": started_at}],
            }
        },
    )
    try:
        while True:
            run_doc = await db.usecase_flow_runs.find_one({"_id": run_id}, {"pause_requested": 1, "stop_requested": 1})
            if not run_doc:
                break
            if run_doc.get("stop_requested"):
                raise asyncio.CancelledError("Run stopped by user.")
            if run_doc.get("pause_requested"):
                await db.usecase_flow_runs.update_one(
                    {"_id": run_id},
                    {"$set": {"status": "paused", "updated_at": _utcnow()}},
                )
                await asyncio.sleep(1)
                continue
            break
        await db.usecase_flow_runs.update_one({"_id": run_id}, {"$set": {"status": "running", "updated_at": _utcnow()}})

        result = await generate_usecase_flow(
            tenant_id=tenant_id,
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=str(run_id),
        )
        run_doc = await db.usecase_flow_runs.find_one({"_id": run_id}, {"stop_requested": 1})
        if run_doc and run_doc.get("stop_requested"):
            raise asyncio.CancelledError("Run stopped by user.")

        saved = await _save_usecase_flow(
            intake_id=intake_id,
            tenant_id=tenant_id,
            project_id=project_id,
            result=result,
        )
        completed_at = _utcnow()
        await db.usecase_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "error": None,
                    "result": {
                        **result,
                        "version": saved.get("version"),
                        "generated_at": saved.get("generated_at"),
                    },
                    "steps": [
                        {
                            "stage": "usecase_generation",
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
                        }
                    ],
                }
            },
        )
    except asyncio.CancelledError:
        stopped_at = _utcnow()
        await db.usecase_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "completed_at": stopped_at,
                    "updated_at": stopped_at,
                    "error": "Run stopped by user.",
                    "steps": [
                        {
                            "stage": "usecase_generation",
                            "status": "stopped",
                            "started_at": started_at,
                            "completed_at": stopped_at,
                            "duration_seconds": round((stopped_at - started_at).total_seconds(), 3),
                            "error": "Run stopped by user.",
                        }
                    ],
                }
            },
        )
    except Exception as exc:
        logger.exception("usecase_flow_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        failed_at = _utcnow()
        await db.usecase_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": failed_at,
                    "updated_at": failed_at,
                    "error": str(exc),
                    "steps": [
                        {
                            "stage": "usecase_generation",
                            "status": "failed",
                            "started_at": started_at,
                            "completed_at": failed_at,
                            "duration_seconds": round((failed_at - started_at).total_seconds(), 3),
                            "error": str(exc),
                        }
                    ],
                }
            },
        )
    finally:
        _ACTIVE_USECASE_TASKS_BY_RUN_ID.pop(str(run_id), None)


async def _save_sequence_flow(
    *,
    intake_id: str,
    tenant_id: str,
    project_id: str,
    result: dict,
) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    last = await db.sequence_flow_documents.find_one(
        {
            "intake_id": intake_oid,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
        },
        sort=[("version", -1)],
    )
    next_version = (last.get("version") if last else 0) + 1
    now = _utcnow()
    mermaid = str(result.get("mermaid") or "").strip() or _sequence_to_mermaid(
        result.get("nodes") or [],
        result.get("edges") or [],
    )
    doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": result.get("nodes") or [],
        "edges": result.get("edges") or [],
        "mermaid": mermaid,
        "summary": result.get("summary") or "",
        "updated_at": now,
    }
    await db.sequence_flow_documents.insert_one(doc)
    latest_doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "nodes": doc["nodes"],
        "edges": doc["edges"],
        "mermaid": doc["mermaid"],
        "summary": doc["summary"],
        "updated_at": now,
    }
    await db.sequence_flows.update_one(
        {"intake_id": intake_oid, "tenant_id": tenant_oid, "project_id": project_oid},
        {"$set": latest_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return doc


async def _save_architecture_diagram(
    *,
    intake_id: str,
    tenant_id: str,
    project_id: str,
    result: dict,
) -> dict:
    db = get_db()
    intake_oid = ObjectId(intake_id)
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    last = await db.architecture_diagram_documents.find_one(
        {
            "intake_id": intake_oid,
            "tenant_id": tenant_oid,
            "project_id": project_oid,
        },
        sort=[("version", -1)],
    )
    next_version = (last.get("version") if last else 0) + 1
    now = _utcnow()
    mermaid = str(result.get("mermaid") or "").strip()
    doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "mermaid": mermaid,
        "summary": str(result.get("summary") or "").strip(),
        "view": str(result.get("view") or "").strip(),
        "updated_at": now,
    }
    await db.architecture_diagram_documents.insert_one(doc)
    latest_doc = {
        "intake_id": intake_oid,
        "tenant_id": tenant_oid,
        "project_id": project_oid,
        "version": next_version,
        "generated_at": now,
        "mermaid": mermaid,
        "summary": str(result.get("summary") or "").strip(),
        "view": str(result.get("view") or "").strip(),
        "updated_at": now,
    }
    await db.architecture_diagrams.update_one(
        {"intake_id": intake_oid, "tenant_id": tenant_oid, "project_id": project_oid},
        {"$set": latest_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return doc


async def _run_sequence_flow_job(
    *,
    run_id: ObjectId,
    intake_id: str,
    project_id: str,
    tenant_id: str,
    structured: dict,
    user_request: str,
    current_nodes: list,
    current_edges: list,
    latest_sdd_content: str,
) -> None:
    db = get_db()
    started_at = _utcnow()
    await db.sequence_flow_runs.update_one(
        {"_id": run_id},
        {
            "$set": {
                "status": "running",
                "started_at": started_at,
                "updated_at": started_at,
                "steps": [{"stage": "sequence_generation", "status": "running", "started_at": started_at}],
            }
        },
    )
    try:
        while True:
            run_doc = await db.sequence_flow_runs.find_one({"_id": run_id}, {"pause_requested": 1, "stop_requested": 1})
            if not run_doc:
                break
            if run_doc.get("stop_requested"):
                raise asyncio.CancelledError("Run stopped by user.")
            if run_doc.get("pause_requested"):
                await db.sequence_flow_runs.update_one(
                    {"_id": run_id},
                    {"$set": {"status": "paused", "updated_at": _utcnow()}},
                )
                await asyncio.sleep(1)
                continue
            break
        await db.sequence_flow_runs.update_one({"_id": run_id}, {"$set": {"status": "running", "updated_at": _utcnow()}})

        result = await generate_usecase_flow(
            tenant_id=tenant_id,
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=str(run_id),
        )
        if not str(result.get("mermaid") or "").strip():
            result["mermaid"] = _sequence_to_mermaid(result.get("nodes") or [], result.get("edges") or [])
        run_doc = await db.sequence_flow_runs.find_one({"_id": run_id}, {"stop_requested": 1})
        if run_doc and run_doc.get("stop_requested"):
            raise asyncio.CancelledError("Run stopped by user.")

        saved = await _save_sequence_flow(
            intake_id=intake_id,
            tenant_id=tenant_id,
            project_id=project_id,
            result=result,
        )
        completed_at = _utcnow()
        await db.sequence_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "error": None,
                    "result": {
                        **result,
                        "version": saved.get("version"),
                        "generated_at": saved.get("generated_at"),
                    },
                    "steps": [
                        {
                            "stage": "sequence_generation",
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
                        }
                    ],
                }
            },
        )
    except asyncio.CancelledError:
        stopped_at = _utcnow()
        await db.sequence_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "completed_at": stopped_at,
                    "updated_at": stopped_at,
                    "error": "Run stopped by user.",
                    "steps": [
                        {
                            "stage": "sequence_generation",
                            "status": "stopped",
                            "started_at": started_at,
                            "completed_at": stopped_at,
                            "duration_seconds": round((stopped_at - started_at).total_seconds(), 3),
                            "error": "Run stopped by user.",
                        }
                    ],
                }
            },
        )
    except Exception as exc:
        logger.exception("sequence_flow_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        failed_at = _utcnow()
        await db.sequence_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": failed_at,
                    "updated_at": failed_at,
                    "error": str(exc),
                    "steps": [
                        {
                            "stage": "sequence_generation",
                            "status": "failed",
                            "started_at": started_at,
                            "completed_at": failed_at,
                            "duration_seconds": round((failed_at - started_at).total_seconds(), 3),
                            "error": str(exc),
                        }
                    ],
                }
            },
        )
    finally:
        _ACTIVE_SEQUENCE_TASKS_BY_RUN_ID.pop(str(run_id), None)


async def _run_schema_flow_job(
    *,
    run_id: ObjectId,
    intake_id: str,
    project_id: str,
    tenant_id: str,
    structured: dict,
    user_request: str,
    current_nodes: list,
    current_edges: list,
    latest_sdd_content: str,
) -> None:
    db = get_db()
    started_at = _utcnow()
    await db.schema_flow_runs.update_one(
        {"_id": run_id},
        {
            "$set": {
                "status": "running",
                "started_at": started_at,
                "updated_at": started_at,
                "steps": [{"stage": "schema_generation", "status": "running", "started_at": started_at}],
            }
        },
    )
    try:
        while True:
            run_doc = await db.schema_flow_runs.find_one({"_id": run_id}, {"pause_requested": 1, "stop_requested": 1})
            if not run_doc:
                break
            if run_doc.get("stop_requested"):
                raise asyncio.CancelledError("Run stopped by user.")
            if run_doc.get("pause_requested"):
                await db.schema_flow_runs.update_one(
                    {"_id": run_id},
                    {"$set": {"status": "paused", "updated_at": _utcnow()}},
                )
                await asyncio.sleep(1)
                continue
            break
        await db.schema_flow_runs.update_one(
            {"_id": run_id},
            {"$set": {"status": "running", "updated_at": _utcnow()}},
        )

        result = await generate_schema_flow(
            tenant_id=tenant_id,
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=str(run_id),
        )
        run_doc = await db.schema_flow_runs.find_one({"_id": run_id}, {"stop_requested": 1})
        if run_doc and run_doc.get("stop_requested"):
            raise asyncio.CancelledError("Run stopped by user.")
        saved = await _save_schema_flow(
            intake_id=intake_id,
            tenant_id=tenant_id,
            project_id=project_id,
            result=result,
        )
        completed_at = _utcnow()
        await db.schema_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "error": None,
                    "result": {
                        **result,
                        "version": saved.get("version"),
                        "generated_at": saved.get("generated_at"),
                    },
                    "steps": [
                        {
                            "stage": "schema_generation",
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
                        }
                    ],
                }
            },
        )
    except asyncio.CancelledError:
        stopped_at = _utcnow()
        await db.schema_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "completed_at": stopped_at,
                    "updated_at": stopped_at,
                    "error": "Run stopped by user.",
                    "steps": [
                        {
                            "stage": "schema_generation",
                            "status": "stopped",
                            "started_at": started_at,
                            "completed_at": stopped_at,
                            "duration_seconds": round((stopped_at - started_at).total_seconds(), 3),
                            "error": "Run stopped by user.",
                        }
                    ],
                }
            },
        )
    except Exception as exc:
        logger.exception("schema_flow_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        failed_at = _utcnow()
        await db.schema_flow_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": failed_at,
                    "updated_at": failed_at,
                    "error": str(exc),
                    "steps": [
                        {
                            "stage": "schema_generation",
                            "status": "failed",
                            "started_at": started_at,
                            "completed_at": failed_at,
                            "duration_seconds": round((failed_at - started_at).total_seconds(), 3),
                            "error": str(exc),
                        }
                    ],
                }
            },
        )
    finally:
        _ACTIVE_SCHEMA_TASKS_BY_RUN_ID.pop(str(run_id), None)


@router.post("/start", response_model=RequirementsStartResponse)
async def start_requirements(
    payload: RequirementsStartRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id query parameter is required")
        resolved_project_id = project_id
        result = await start_intake(user.get("tenant_id"), resolved_project_id, payload.raw_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/respond", response_model=RequirementsRespondResponse)
async def respond_requirements(
    payload: RequirementsRespondRequest,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        result = await respond_intake(payload.intake_id, payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/generate", response_model=RequirementsGenerateResponse)
async def generate_requirements(
    payload: RequirementsGenerateRequest,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        result = await generate_prd(payload.intake_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"prd": result}


@router.post("/{intake_id}/generate-prd")
async def generate_full_prd(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    if missing or low_quality:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requirement not complete. Cannot generate PRD.",
                "missing_fields": missing,
                "low_quality_fields": low_quality,
            },
        )
    structured = intake.get("structured") or {}
    try:
        content = generate_prd_text(structured)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    saved = await save_prd(
        intake_id,
        content,
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )
    return {"content": content, "version": saved["version"]}


@router.post("/{intake_id}/generate-system-design")
async def generate_system_design_doc(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    if missing or low_quality:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requirement not complete. Cannot generate system design.",
                "missing_fields": missing,
                "low_quality_fields": low_quality,
            },
        )
    structured = intake.get("structured") or {}
    content = await generate_system_design(
        structured,
        _user.get("tenant_id"),
        project_id=project_id,
        intake_id=intake_id,
        run_id=None,
    )
    saved = await _save_system_design(intake_id, project_id, _user.get("tenant_id"), content)
    return {"content": content, "version": saved["version"]}


@router.post("/{intake_id}/generate-system-design/run")
async def generate_system_design_run(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    if missing or low_quality:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requirement not complete. Cannot generate system design.",
                "missing_fields": missing,
                "low_quality_fields": low_quality,
            },
        )
    structured = intake.get("structured") or {}

    run_id = ObjectId()
    created_at = _utcnow()
    await db.sdd_runs.insert_one(
        {
            "_id": run_id,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "intake_id": ObjectId(intake_id),
            "status": "queued",
            "pause_requested": False,
            "stop_requested": False,
            "steps": [],
            "error": None,
            "result": None,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
        }
    )

    task = asyncio.create_task(
        _run_sdd_job(
            run_id=run_id,
            intake_id=intake_id,
            project_id=project_id,
            tenant_id=user.get("tenant_id"),
            structured=structured,
        )
    )
    _ACTIVE_SDD_TASKS_BY_RUN_ID[str(run_id)] = task
    return {"run_id": str(run_id), "status": "queued"}


@router.get("/system-design/runs/{run_id}")
async def get_system_design_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sdd_runs.find_one(
        {
            "_id": oid,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="SDD run not found")
    total_elapsed = None
    started_at = _coerce_utc_datetime(doc.get("started_at"))
    if started_at:
        end_time = _coerce_utc_datetime(doc.get("completed_at")) or _utcnow()
        total_elapsed = round((end_time - started_at).total_seconds(), 3)
    return {
        "run_id": str(doc.get("_id")),
        "status": doc.get("status"),
        "steps": doc.get("steps") or [],
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {"total_elapsed_seconds": total_elapsed},
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/system-design/runs/{run_id}/pause")
async def pause_system_design_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sdd_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="SDD run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.sdd_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": True, "status": "paused", "updated_at": _utcnow()}},
    )
    await _append_sdd_run_event(oid, {"stage": "run", "status": "paused"})
    return {"run_id": run_id, "status": "paused"}


@router.post("/system-design/runs/{run_id}/resume")
async def resume_system_design_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sdd_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="SDD run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.sdd_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": False, "status": "running", "updated_at": _utcnow()}},
    )
    await _append_sdd_run_event(oid, {"stage": "run", "status": "running"})
    return {"run_id": run_id, "status": "running"}


@router.post("/system-design/runs/{run_id}/stop")
async def stop_system_design_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sdd_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="SDD run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    now = _utcnow()
    await db.sdd_runs.update_one(
        {"_id": oid},
        {
            "$set": {
                "stop_requested": True,
                "pause_requested": False,
                "status": "stopped",
                "completed_at": now,
                "updated_at": now,
                "error": "Run stopped by user.",
            }
        },
    )
    task = _ACTIVE_SDD_TASKS_BY_RUN_ID.get(run_id)
    if task and not task.done():
        task.cancel()
    await _append_sdd_run_event(oid, {"stage": "run", "status": "stopped", "error": "Run stopped by user."})
    return {"run_id": run_id, "status": "stopped"}


@router.get("/{intake_id}/system-design")
async def get_system_design_doc(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    if not ObjectId.is_valid(intake_id):
        if intake_id == "latest":
            doc = await db.system_design_documents.find_one(
                {
                    "tenant_id": ObjectId(_user.get("tenant_id")),
                    "project_id": ObjectId(project_id),
                },
                sort=[("generated_at", -1)],
            )
            if not doc:
                raise HTTPException(status_code=404, detail="System design not found")
            return {
                "content": doc.get("content"),
                "version": doc.get("version"),
                "generated_at": doc.get("generated_at"),
            }
        raise HTTPException(status_code=400, detail="Invalid intake_id")

    doc = await db.system_design_documents.find_one(
        {"intake_id": ObjectId(intake_id)},
        sort=[("version", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="System design not found")
    return {
        "content": doc.get("content"),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
    }


@router.get("/latest/system-design")
async def get_latest_system_design_doc(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.system_design_documents.find_one(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        sort=[("generated_at", -1)],
    )
    latest_run = await db.sdd_runs.find_one(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        sort=[("created_at", -1)],
    )
    if not doc and not latest_run:
        raise HTTPException(status_code=404, detail="System design not found")
    return {
        "content": doc.get("content") if doc else None,
        "version": doc.get("version") if doc else None,
        "generated_at": doc.get("generated_at") if doc else None,
        "run_id": str(latest_run.get("_id")) if latest_run else None,
        "run_status": latest_run.get("status") if latest_run else None,
        "steps": latest_run.get("steps") if latest_run else [],
        "run_error": latest_run.get("error") if latest_run else None,
    }


@router.get("/system-design/versions")
async def get_system_design_versions(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.system_design_documents.find(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "generated_at": 1, "intake_id": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": doc.get("version"),
                "generated_at": doc.get("generated_at"),
                "intake_id": str(doc.get("intake_id")) if doc.get("intake_id") else None,
            }
            for doc in docs
        ]
    }


@router.get("/system-design/versions/{version_number}")
async def get_system_design_by_version(
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.system_design_documents.find_one(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        },
        sort=[("generated_at", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="System design version not found")
    return {
        "content": doc.get("content"),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
    }


@router.get("/latest")
async def get_latest_requirements_status(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")

    db = get_db()
    intake = await db.requirements_intakes.find_one(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        sort=[("updated_at", -1), ("created_at", -1)],
    )
    if not intake:
        raise HTTPException(status_code=404, detail="No requirements intake found")

    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    return {
        "intake_id": str(intake["_id"]),
        "structured_partial": intake.get("structured") or {},
        "missing_fields": missing,
        "low_quality_fields": low_quality,
        "questions": intake.get("questions") or [],
        "ready_for_prd": len(missing) == 0 and len(low_quality) == 0,
    }


@router.get("/{intake_id}")
async def get_requirements_status(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    return {
        "structured_partial": intake.get("structured") or {},
        "missing_fields": missing,
        "low_quality_fields": low_quality,
        "questions": intake.get("questions") or [],
        "ready_for_prd": len(missing) == 0 and len(low_quality) == 0,
    }


@router.get("/{intake_id}/prd")
async def get_prd(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.prd_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        sort=[("version", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="PRD not found")
    return {
        "content": doc.get("content"),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
    }


@router.get("/{intake_id}/prd/versions")
async def get_prd_versions(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.prd_documents.find(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "generated_at": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": d.get("version"),
                "generated_at": d.get("generated_at"),
            }
            for d in docs
        ]
    }


@router.get("/{intake_id}/prd/versions/{version_number}")
async def get_prd_version(
    intake_id: str,
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.prd_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="PRD version not found")
    return {
        "content": doc.get("content"),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
    }


@router.post("/{intake_id}/generate-schema-flow")
async def generate_schema_flow_doc(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        raise HTTPException(status_code=400, detail="request is required")
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )
    try:
        result = await generate_schema_flow(
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    saved = await _save_schema_flow(
        intake_id=intake_id,
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        result=result,
    )
    return {
        **result,
        "version": saved.get("version"),
        "generated_at": saved.get("generated_at"),
    }


@router.post("/{intake_id}/generate-schema-flow/run")
async def generate_schema_flow_run(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")

    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        user_request = "Generate initial database schema plan from requirements and architecture inputs."
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )

    run_id = ObjectId()
    created_at = _utcnow()
    await db.schema_flow_runs.insert_one(
        {
            "_id": run_id,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "intake_id": ObjectId(intake_id),
            "status": "queued",
            "pause_requested": False,
            "stop_requested": False,
            "steps": [],
            "error": None,
            "result": None,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "request": user_request,
        }
    )

    task = asyncio.create_task(
        _run_schema_flow_job(
            run_id=run_id,
            intake_id=intake_id,
            project_id=project_id,
            tenant_id=user.get("tenant_id"),
            structured=structured,
            user_request=user_request,
            current_nodes=current_nodes,
            current_edges=current_edges,
            latest_sdd_content=latest_sdd_content,
        )
    )
    _ACTIVE_SCHEMA_TASKS_BY_RUN_ID[str(run_id)] = task
    return {"run_id": str(run_id), "status": "queued"}


@router.get("/schema-flow/runs/{run_id}")
async def get_schema_flow_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.schema_flow_runs.find_one(
        {
            "_id": oid,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Schema flow run not found")

    total_elapsed = None
    started_at = _coerce_utc_datetime(doc.get("started_at"))
    if started_at:
        end_time = _coerce_utc_datetime(doc.get("completed_at")) or _utcnow()
        total_elapsed = round((end_time - started_at).total_seconds(), 3)

    return {
        "run_id": str(doc.get("_id")),
        "status": doc.get("status"),
        "steps": doc.get("steps") or [],
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {"total_elapsed_seconds": total_elapsed},
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/schema-flow/runs/{run_id}/pause")
async def pause_schema_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.schema_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Schema flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.schema_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": True, "status": "paused", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "paused"}


@router.post("/schema-flow/runs/{run_id}/resume")
async def resume_schema_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.schema_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Schema flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.schema_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": False, "status": "running", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "running"}


@router.post("/schema-flow/runs/{run_id}/stop")
async def stop_schema_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.schema_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Schema flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    now = _utcnow()
    await db.schema_flow_runs.update_one(
        {"_id": oid},
        {
            "$set": {
                "stop_requested": True,
                "pause_requested": False,
                "status": "stopped",
                "completed_at": now,
                "updated_at": now,
                "error": "Run stopped by user.",
            }
        },
    )
    task = _ACTIVE_SCHEMA_TASKS_BY_RUN_ID.get(run_id)
    if task and not task.done():
        task.cancel()
    return {"run_id": run_id, "status": "stopped"}


@router.get("/{intake_id}/schema-flow")
async def get_schema_flow_doc(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.schema_flows.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        return {
            "nodes": [],
            "edges": [],
            "summary": "",
            "updated_at": None,
            "exists": False,
        }
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.get("/{intake_id}/schema-flow/versions")
async def get_schema_flow_versions(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.schema_flow_documents.find(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "generated_at": 1, "updated_at": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": d.get("version"),
                "generated_at": d.get("generated_at"),
                "updated_at": d.get("updated_at"),
            }
            for d in docs
        ]
    }


@router.get("/{intake_id}/schema-flow/versions/{version_number}")
async def get_schema_flow_version(
    intake_id: str,
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.schema_flow_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Schema flow version not found")
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.post("/{intake_id}/generate-usecase-flow")
async def generate_usecase_flow_doc(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        raise HTTPException(status_code=400, detail="request is required")
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )
    try:
        result = await generate_usecase_flow(
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=None,
        )
        if not str(result.get("mermaid") or "").strip():
            result["mermaid"] = _sequence_to_mermaid(result.get("nodes") or [], result.get("edges") or [])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    saved = await _save_usecase_flow(
        intake_id=intake_id,
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        result=result,
    )
    return {
        **result,
        "version": saved.get("version"),
        "generated_at": saved.get("generated_at"),
    }


@router.post("/{intake_id}/generate-usecase-flow/run")
async def generate_usecase_flow_run(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")

    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        user_request = "Generate initial use case interaction diagram from requirements and architecture inputs."
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )

    run_id = ObjectId()
    created_at = _utcnow()
    await db.usecase_flow_runs.insert_one(
        {
            "_id": run_id,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "intake_id": ObjectId(intake_id),
            "status": "queued",
            "pause_requested": False,
            "stop_requested": False,
            "steps": [],
            "error": None,
            "result": None,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "request": user_request,
        }
    )
    task = asyncio.create_task(
        _run_usecase_flow_job(
            run_id=run_id,
            intake_id=intake_id,
            project_id=project_id,
            tenant_id=user.get("tenant_id"),
            structured=structured,
            user_request=user_request,
            current_nodes=current_nodes,
            current_edges=current_edges,
            latest_sdd_content=latest_sdd_content,
        )
    )
    _ACTIVE_USECASE_TASKS_BY_RUN_ID[str(run_id)] = task
    return {"run_id": str(run_id), "status": "queued"}


@router.get("/usecase-flow/runs/{run_id}")
async def get_usecase_flow_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.usecase_flow_runs.find_one(
        {
            "_id": oid,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usecase flow run not found")
    total_elapsed = None
    started_at = _coerce_utc_datetime(doc.get("started_at"))
    if started_at:
        end_time = _coerce_utc_datetime(doc.get("completed_at")) or _utcnow()
        total_elapsed = round((end_time - started_at).total_seconds(), 3)
    return {
        "run_id": str(doc.get("_id")),
        "status": doc.get("status"),
        "steps": doc.get("steps") or [],
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {"total_elapsed_seconds": total_elapsed},
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/usecase-flow/runs/{run_id}/pause")
async def pause_usecase_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.usecase_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usecase flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.usecase_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": True, "status": "paused", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "paused"}


@router.post("/usecase-flow/runs/{run_id}/resume")
async def resume_usecase_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.usecase_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usecase flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.usecase_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": False, "status": "running", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "running"}


@router.post("/usecase-flow/runs/{run_id}/stop")
async def stop_usecase_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.usecase_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usecase flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    now = _utcnow()
    await db.usecase_flow_runs.update_one(
        {"_id": oid},
        {
            "$set": {
                "stop_requested": True,
                "pause_requested": False,
                "status": "stopped",
                "completed_at": now,
                "updated_at": now,
                "error": "Run stopped by user.",
            }
        },
    )
    task = _ACTIVE_USECASE_TASKS_BY_RUN_ID.get(run_id)
    if task and not task.done():
        task.cancel()
    return {"run_id": run_id, "status": "stopped"}


@router.get("/{intake_id}/usecase-flow")
async def get_usecase_flow_doc(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.usecase_flows.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        return {"nodes": [], "edges": [], "mermaid": "sequenceDiagram", "summary": "", "updated_at": None, "exists": False}
    mermaid = str(doc.get("mermaid") or "").strip() or _sequence_to_mermaid(doc.get("nodes") or [], doc.get("edges") or [])
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "mermaid": mermaid,
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.get("/{intake_id}/usecase-flow/versions")
async def get_usecase_flow_versions(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.usecase_flow_documents.find(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "generated_at": 1, "updated_at": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": d.get("version"),
                "generated_at": d.get("generated_at"),
                "updated_at": d.get("updated_at"),
            }
            for d in docs
        ]
    }


@router.get("/{intake_id}/usecase-flow/versions/{version_number}")
async def get_usecase_flow_version(
    intake_id: str,
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.usecase_flow_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Usecase flow version not found")
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.post("/{intake_id}/generate-sequence-flow")
async def generate_sequence_flow_doc(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        raise HTTPException(status_code=400, detail="request is required")
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )
    try:
        result = await generate_usecase_flow(
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            intake_id=intake_id,
            structured=structured,
            current_nodes=current_nodes,
            current_edges=current_edges,
            user_request=user_request,
            latest_sdd_content=latest_sdd_content,
            run_id=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    saved = await _save_sequence_flow(
        intake_id=intake_id,
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        result=result,
    )
    return {
        **result,
        "version": saved.get("version"),
        "generated_at": saved.get("generated_at"),
    }


@router.post("/{intake_id}/generate-sequence-flow/run")
async def generate_sequence_flow_run(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")

    structured = intake.get("structured") or {}
    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        user_request = "Generate initial sequence interaction diagram from requirements and architecture inputs."
    current_nodes = payload.get("nodes") or []
    current_edges = payload.get("edges") or []
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )

    run_id = ObjectId()
    created_at = _utcnow()
    await db.sequence_flow_runs.insert_one(
        {
            "_id": run_id,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "intake_id": ObjectId(intake_id),
            "status": "queued",
            "pause_requested": False,
            "stop_requested": False,
            "steps": [],
            "error": None,
            "result": None,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "request": user_request,
        }
    )
    task = asyncio.create_task(
        _run_sequence_flow_job(
            run_id=run_id,
            intake_id=intake_id,
            project_id=project_id,
            tenant_id=user.get("tenant_id"),
            structured=structured,
            user_request=user_request,
            current_nodes=current_nodes,
            current_edges=current_edges,
            latest_sdd_content=latest_sdd_content,
        )
    )
    _ACTIVE_SEQUENCE_TASKS_BY_RUN_ID[str(run_id)] = task
    return {"run_id": str(run_id), "status": "queued"}


@router.get("/sequence-flow/runs/{run_id}")
async def get_sequence_flow_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sequence_flow_runs.find_one(
        {
            "_id": oid,
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sequence flow run not found")
    total_elapsed = None
    started_at = _coerce_utc_datetime(doc.get("started_at"))
    if started_at:
        end_time = _coerce_utc_datetime(doc.get("completed_at")) or _utcnow()
        total_elapsed = round((end_time - started_at).total_seconds(), 3)
    return {
        "run_id": str(doc.get("_id")),
        "status": doc.get("status"),
        "steps": doc.get("steps") or [],
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {"total_elapsed_seconds": total_elapsed},
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/sequence-flow/runs/{run_id}/pause")
async def pause_sequence_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sequence_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sequence flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.sequence_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": True, "status": "paused", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "paused"}


@router.post("/sequence-flow/runs/{run_id}/resume")
async def resume_sequence_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sequence_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sequence flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.sequence_flow_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": False, "status": "running", "updated_at": _utcnow()}},
    )
    return {"run_id": run_id, "status": "running"}


@router.post("/sequence-flow/runs/{run_id}/stop")
async def stop_sequence_flow_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.sequence_flow_runs.find_one(
        {"_id": oid, "tenant_id": ObjectId(user.get("tenant_id")), "project_id": ObjectId(project_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sequence flow run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    now = _utcnow()
    await db.sequence_flow_runs.update_one(
        {"_id": oid},
        {
            "$set": {
                "stop_requested": True,
                "pause_requested": False,
                "status": "stopped",
                "completed_at": now,
                "updated_at": now,
                "error": "Run stopped by user.",
            }
        },
    )
    task = _ACTIVE_SEQUENCE_TASKS_BY_RUN_ID.get(run_id)
    if task and not task.done():
        task.cancel()
    return {"run_id": run_id, "status": "stopped"}


@router.get("/{intake_id}/sequence-flow")
async def get_sequence_flow_doc(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.sequence_flows.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        return {"nodes": [], "edges": [], "summary": "", "updated_at": None, "exists": False}
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.get("/{intake_id}/sequence-flow/versions")
async def get_sequence_flow_versions(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.sequence_flow_documents.find(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "generated_at": 1, "updated_at": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": d.get("version"),
                "generated_at": d.get("generated_at"),
                "updated_at": d.get("updated_at"),
            }
            for d in docs
        ]
    }


@router.get("/{intake_id}/sequence-flow/versions/{version_number}")
async def get_sequence_flow_version(
    intake_id: str,
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.sequence_flow_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Sequence flow version not found")
    mermaid = str(doc.get("mermaid") or "").strip() or _sequence_to_mermaid(doc.get("nodes") or [], doc.get("edges") or [])
    return {
        "nodes": doc.get("nodes") or [],
        "edges": doc.get("edges") or [],
        "mermaid": mermaid,
        "summary": doc.get("summary") or "",
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.post("/{intake_id}/generate-architecture-diagram")
async def generate_architecture_diagram_doc(
    intake_id: str,
    payload: dict,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    structured = intake.get("structured") or {}

    user_request = str(payload.get("request") or "").strip()
    if not user_request:
        user_request = "Show the high-level system architecture with Load Balancer and AWS components."
    latest_sdd_content = await _get_latest_sdd_content(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
    )

    try:
        result = await generate_architecture_mermaid(
            user_request=user_request,
            structured=structured,
            latest_sdd_content=latest_sdd_content,
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            intake_id=intake_id,
            run_id=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    saved = await _save_architecture_diagram(
        intake_id=intake_id,
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        result=result,
    )
    return {
        "mermaid": result.get("mermaid") or "",
        "summary": result.get("summary") or "",
        "view": result.get("view") or "",
        "version": saved.get("version"),
        "generated_at": saved.get("generated_at"),
        "updated_at": saved.get("updated_at"),
        "exists": True,
    }


@router.get("/{intake_id}/architecture-diagram")
async def get_architecture_diagram_doc(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.architecture_diagrams.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        }
    )
    if not doc:
        return {
            "mermaid": "",
            "summary": "",
            "view": "",
            "version": None,
            "generated_at": None,
            "updated_at": None,
            "exists": False,
        }
    return {
        "mermaid": str(doc.get("mermaid") or ""),
        "summary": str(doc.get("summary") or ""),
        "view": str(doc.get("view") or ""),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.get("/{intake_id}/architecture-diagram/versions")
async def get_architecture_diagram_versions(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    docs = await db.architecture_diagram_documents.find(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
        },
        {"version": 1, "view": 1, "generated_at": 1, "updated_at": 1},
    ).sort("version", -1).to_list(length=200)
    return {
        "items": [
            {
                "version_number": d.get("version"),
                "view": d.get("view"),
                "generated_at": d.get("generated_at"),
                "updated_at": d.get("updated_at"),
            }
            for d in docs
        ]
    }


@router.get("/{intake_id}/architecture-diagram/versions/{version_number}")
async def get_architecture_diagram_version(
    intake_id: str,
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.architecture_diagram_documents.find_one(
        {
            "intake_id": ObjectId(intake_id),
            "tenant_id": ObjectId(user.get("tenant_id")),
            "project_id": ObjectId(project_id),
            "version": version_number,
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Architecture diagram version not found")
    return {
        "mermaid": str(doc.get("mermaid") or ""),
        "summary": str(doc.get("summary") or ""),
        "view": str(doc.get("view") or ""),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
        "updated_at": doc.get("updated_at"),
        "exists": True,
    }


@router.post("/undo")
async def undo_requirements(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        structured = await undo_intake(user.get("tenant_id"), project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"structured_partial": structured}


@router.post("/redo")
async def redo_requirements(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        structured = await redo_intake(user.get("tenant_id"), project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"structured_partial": structured}
