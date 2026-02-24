from datetime import datetime, timedelta, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.guard import withGuard
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate
from app.services.audit_service import log_event
from app.services.project_member_service import add_project_member
from app.services.project_service import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    restore_project,
    update_project,
)
from app.db.mongo import get_db
from app.core.config import settings


router = APIRouter(prefix="/api/projects", tags=["projects"])


def _json_safe(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _normalize(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


@router.get("", response_model=list[ProjectOut])
async def list_projects_route(
    request: Request,
    q: str | None = None,
    status: str | None = None,
    user=Depends(withGuard(feature="view_decision", orgRole="viewer")),
):
    db = get_db()
    memberships = await db.project_members.find(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "user_id": ObjectId(user.get("user_id")),
            "deleted_at": None,
        }
    ).to_list(length=500)
    project_ids = [str(doc["project_id"]) for doc in memberships]
    if not project_ids:
        return []
    projects = await list_projects(
        user.get("tenant_id"),
        project_ids=project_ids,
        search=q,
        status=status,
    )
    return [_normalize(doc) for doc in projects]


@router.post("", response_model=ProjectOut)
async def create_project_route(
    payload: ProjectCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    project = await create_project(request.state.tenant_id, payload.model_dump())
    project_id = project.get("_id") or project.get("id")
    await add_project_member(request.state.tenant_id, project_id, user.get("user_id"), "project_admin")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.created",
        entity_type="project",
        entity_id=project_id,
    )
    return _normalize(project)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project_route(
    project_id: str,
    request: Request,
    _guard=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    project = await get_project(request.state.tenant_id, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return _normalize(project)


@router.get("/{project_id}/dashboard/owner-summary")
async def get_owner_dashboard_summary(
    project_id: str,
    request: Request,
    days: int = 7,
    _guard=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")
    db = get_db()
    tenant_id = request.state.tenant_id
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)
    day_ago = now - timedelta(days=1)

    project = await db.projects.find_one(
        {"_id": project_oid, "tenant_id": tenant_oid},
        {"name": 1, "status": 1, "created_at": 1, "updated_at": 1, "description": 1},
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    members_count = await db.project_members.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "deleted_at": None}
    )
    channels_count = await db.project_channels.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "deleted_at": None}
    )
    threads_count = await db.project_threads.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "deleted_at": None}
    )
    messages_count = await db.project_messages.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid}
    )
    messages_24h = await db.project_messages.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "created_at": {"$gte": day_ago}}
    )
    personal_chats_count = await db.project_personal_chats.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "deleted_at": None}
    )

    decisions_count = await db.decisions.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid}
    )
    decisions_7d = await db.decisions.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "created_at": {"$gte": window_start}}
    )
    recent_decisions_cursor = db.decisions.find(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        {"title": 1, "statement": 1, "source": 1, "created_at": 1, "timestamp": 1},
    ).sort("created_at", -1).limit(8)
    recent_decisions = []
    async for doc in recent_decisions_cursor:
        recent_decisions.append(
            {
                "id": str(doc["_id"]),
                "title": doc.get("title") or "Decision",
                "statement": doc.get("statement") or "",
                "source": doc.get("source") or "unknown",
                "created_at": doc.get("created_at") or doc.get("timestamp"),
            }
        )

    requirements_latest = await db.requirements_intakes.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        {"status": 1, "created_at": 1, "updated_at": 1},
        sort=[("created_at", -1)],
    )
    prd_latest_run = await db.prd_runs.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        {"status": 1, "created_at": 1, "updated_at": 1, "completed_at": 1, "result": 1},
        sort=[("created_at", -1)],
    )
    active_prd_runs = await db.prd_runs.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "status": {"$in": ["queued", "running", "paused"]}}
    )
    prd_runs_7d = await db.prd_runs.count_documents(
        {"tenant_id": tenant_oid, "project_id": project_oid, "created_at": {"$gte": window_start}}
    )

    llm_match = {
        "created_at": {"$gte": window_start},
        "tenant_id": {"$in": [tenant_id, tenant_oid]},
    }
    llm_overall = await db.llm_usage_logs.aggregate(
        [
            {"$match": llm_match},
            {
                "$group": {
                    "_id": None,
                    "requests": {"$sum": 1},
                    "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
                    "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
                    "estimated_cost": {"$sum": {"$ifNull": ["$estimated_cost", 0.0]}},
                    "max_input_tokens": {"$max": {"$ifNull": ["$input_tokens", 0]}},
                    "max_output_tokens": {"$max": {"$ifNull": ["$output_tokens", 0]}},
                }
            },
        ]
    ).to_list(length=1)

    llm_feature_rows = await db.llm_usage_logs.aggregate(
        [
            {"$match": llm_match},
            {
                "$group": {
                    "_id": {"$ifNull": ["$feature", "unknown"]},
                    "requests": {"$sum": 1},
                    "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
                    "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
                    "estimated_cost": {"$sum": {"$ifNull": ["$estimated_cost", 0.0]}},
                }
            },
            {"$sort": {"input_tokens": -1, "output_tokens": -1}},
            {"$limit": 6},
        ]
    ).to_list(length=6)

    llm_daily_rows = await db.llm_usage_logs.aggregate(
        [
            {"$match": llm_match},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"},
                    },
                    "requests": {"$sum": 1},
                    "tokens": {
                        "$sum": {
                            "$add": [
                                {"$ifNull": ["$input_tokens", 0]},
                                {"$ifNull": ["$output_tokens", 0]},
                            ]
                        }
                    },
                    "estimated_cost": {"$sum": {"$ifNull": ["$estimated_cost", 0.0]}},
                }
            },
            {"$sort": {"_id": 1}},
        ]
    ).to_list(length=days + 5)

    llm = llm_overall[0] if llm_overall else {}
    llm_input = int(llm.get("input_tokens") or 0)
    llm_output = int(llm.get("output_tokens") or 0)
    llm_total = llm_input + llm_output
    llm_requests = int(llm.get("requests") or 0)
    llm_max_request_tokens = int(llm.get("max_input_tokens") or 0) + int(llm.get("max_output_tokens") or 0)
    per_request_budget = int(settings.llm_max_input_tokens) + int(settings.llm_max_output_tokens)
    token_headroom_percent = 100
    if per_request_budget > 0:
        token_headroom_percent = max(0, min(100, round(100 - ((llm_max_request_tokens / per_request_budget) * 100), 2)))

    recent_activity_query = {
        "tenant_id": tenant_oid,
        "$or": [
            {"entity_id": project_id},
            {"metadata.project_id": project_id},
            {"action": {"$regex": "project|prd|requirements|decision", "$options": "i"}},
        ],
    }
    recent_activity_docs = await db.audit_logs.find(
        recent_activity_query,
        {"action": 1, "entity_type": 1, "entity_id": 1, "actor_id": 1, "created_at": 1, "metadata": 1},
    ).sort("_id", -1).limit(12).to_list(length=12)
    recent_activity = [
        {
            "id": str(doc["_id"]),
            "action": doc.get("action") or "event",
            "entity_type": doc.get("entity_type") or "unknown",
            "entity_id": _json_safe(doc.get("entity_id")) or "",
            "actor_id": str(doc.get("actor_id")) if doc.get("actor_id") else "",
            "created_at": doc.get("created_at"),
            "metadata": _json_safe(doc.get("metadata") or {}),
        }
        for doc in recent_activity_docs
    ]

    return {
        "window_days": days,
        "project": {
            "id": str(project["_id"]),
            "name": project.get("name") or "Project",
            "description": project.get("description") or "",
            "status": project.get("status") or "active",
            "created_at": project.get("created_at"),
            "updated_at": project.get("updated_at"),
        },
        "kpis": {
            "members": members_count,
            "channels": channels_count,
            "threads": threads_count,
            "messages_total": messages_count,
            "messages_24h": messages_24h,
            "personal_chats": personal_chats_count,
            "decisions_total": decisions_count,
            "decisions_window": decisions_7d,
            "prd_runs_window": prd_runs_7d,
            "active_prd_runs": active_prd_runs,
        },
        "requirements": {
            "status": requirements_latest.get("status") if requirements_latest else None,
            "created_at": requirements_latest.get("created_at") if requirements_latest else None,
            "updated_at": requirements_latest.get("updated_at") if requirements_latest else None,
        },
        "prd": {
            "latest_status": prd_latest_run.get("status") if prd_latest_run else None,
            "latest_created_at": prd_latest_run.get("created_at") if prd_latest_run else None,
            "latest_updated_at": prd_latest_run.get("updated_at") if prd_latest_run else None,
            "latest_completed_at": prd_latest_run.get("completed_at") if prd_latest_run else None,
            "latest_version": (prd_latest_run.get("result") or {}).get("version") if prd_latest_run else None,
        },
        "llm_usage": {
            "window_days": days,
            "requests": llm_requests,
            "input_tokens": llm_input,
            "output_tokens": llm_output,
            "total_tokens": llm_total,
            "estimated_cost": round(float(llm.get("estimated_cost") or 0.0), 6),
            "avg_tokens_per_request": round(llm_total / llm_requests, 2) if llm_requests else 0,
            "max_tokens_per_request": llm_max_request_tokens,
            "token_budget_per_request": per_request_budget,
            "token_headroom_percent": token_headroom_percent,
            "by_feature": [
                {
                    "feature": row.get("_id") or "unknown",
                    "requests": int(row.get("requests") or 0),
                    "input_tokens": int(row.get("input_tokens") or 0),
                    "output_tokens": int(row.get("output_tokens") or 0),
                    "total_tokens": int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0),
                    "estimated_cost": round(float(row.get("estimated_cost") or 0.0), 6),
                }
                for row in llm_feature_rows
            ],
            "daily": [
                {
                    "date": row.get("_id"),
                    "requests": int(row.get("requests") or 0),
                    "tokens": int(row.get("tokens") or 0),
                    "estimated_cost": round(float(row.get("estimated_cost") or 0.0), 6),
                }
                for row in llm_daily_rows
            ],
        },
        "recent_decisions": recent_decisions,
        "recent_activity": recent_activity,
    }


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project_route(
    project_id: str,
    payload: ProjectUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    updated = await update_project(request.state.tenant_id, project_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.updated",
        entity_type="project",
        entity_id=project_id,
    )
    return _normalize(updated)


@router.delete("/{project_id}")
async def delete_project_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    deleted = await delete_project(request.state.tenant_id, project_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.deleted",
        entity_type="project",
        entity_id=project_id,
    )
    return {"status": "deleted"}


@router.post("/{project_id}/restore")
async def restore_project_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    restored, reason = await restore_project(request.state.tenant_id, project_id)
    if not restored:
        if reason == "Project not found":
            raise HTTPException(status_code=404, detail=reason)
        raise HTTPException(status_code=400, detail=reason)
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.restored",
        entity_type="project",
        entity_id=project_id,
    )
    return {"status": "restored"}
