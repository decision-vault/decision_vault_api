"""Owner dashboard aggregation service.

Builds the enriched ``owner-summary`` payload for the project dashboard from
real collection aggregates (``usage_daily``, ``canvases``, ``documents``,
``sprints``, ``workflow_tasks``, ``audit_logs``, ``notifications``,
``feedback``) instead of static zeros. Kept separate from the route so the
aggregation is unit-testable and reusable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from app.db.mongo import get_db
from app.services.license_service import compute_usage
from app.core.plans import get_plan, plan_quota

logger = logging.getLogger("decisionvault.dashboard")

# Rough USD pricing per 1M tokens (Gemini-flavored defaults for display).
INPUT_RATE_PER_1M = 1.25
OUTPUT_RATE_PER_1M = 5.00


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str | Any) -> ObjectId:
    return ObjectId(value)


def _json_safe(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens * INPUT_RATE_PER_1M + completion_tokens * OUTPUT_RATE_PER_1M) / 1_000_000,
        4,
    )


async def _aggregate_usage_window(db, tenant_oid: ObjectId, *, start, end, project_oid: ObjectId | None) -> dict:
    match: dict = {
        "tenant_id": tenant_oid,
        "date": {"$gte": start.strftime("%Y-%m-%d"), "$lt": end.strftime("%Y-%m-%d")},
    }
    if project_oid:
        match["project_id"] = project_oid
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "prompt_tokens": {"$sum": "$prompt_tokens"},
                "completion_tokens": {"$sum": "$completion_tokens"},
                "ai_calls": {"$sum": "$ai_calls"},
            }
        },
    ]
    agg = await db.usage_daily.aggregate(pipeline).to_list(1)
    return agg[0] if agg else {}


async def _usage_series(db, tenant_oid: ObjectId, *, start, end, project_oid: ObjectId | None) -> list[dict]:
    """Daily token/call series for [start, end), zero-filled so charts render a continuous axis."""
    match: dict = {
        "tenant_id": tenant_oid,
        "date": {"$gte": start.strftime("%Y-%m-%d"), "$lt": end.strftime("%Y-%m-%d")},
    }
    if project_oid:
        match["project_id"] = project_oid
    rows = {doc["date"]: doc for doc in await db.usage_daily.find(match).to_list(1000)}
    series: list[dict] = []
    day = start.date()
    while day < end.date():
        key = day.isoformat()
        doc = rows.get(key, {})
        series.append(
            {
                "date": key,
                "total_tokens": int(doc.get("prompt_tokens", 0)) + int(doc.get("completion_tokens", 0)),
                "ai_calls": int(doc.get("ai_calls", 0)),
            }
        )
        day += timedelta(days=1)
    return series


async def build_owner_dashboard_summary(
    tenant_id: str, project_id: str, days: int
) -> dict:
    """Build the full enriched owner-summary payload for one project + window."""
    db = get_db()
    tenant_oid = _oid(tenant_id)
    project_oid = _oid(project_id)
    now = _utcnow()
    window_start = now - timedelta(days=days)
    prev_window_end = window_start
    prev_window_start = prev_window_end - timedelta(days=days)

    project = await db.projects.find_one(
        {"_id": project_oid, "tenant_id": tenant_oid},
        {"name": 1, "status": 1, "created_at": 1, "updated_at": 1, "description": 1},
    )
    if not project:
        raise LookupError("Project not found")

    # ---- Team / capacity ----
    members_count = await db.users.count_documents(
        {"tenant_id": tenant_oid, "deleted_at": None, "is_active": True}
    )
    license_doc = await db.licenses.find_one({"tenant_id": tenant_oid})
    plan_id = (license_doc or {}).get("plan", "free")
    plan = get_plan(plan_id) or {"name": plan_id.title()}
    capacity = await compute_usage(tenant_id, plan_id)
    projects_limit = plan_quota(plan_id, "projects")
    members_limit = plan_quota(plan_id, "team_members")
    ai_limit = plan_quota(plan_id, "ai_tokens")

    # ---- Usage window + previous-window delta ----
    current_agg = await _aggregate_usage_window(
        db, tenant_oid, start=window_start, end=now, project_oid=project_oid
    )
    previous_agg = await _aggregate_usage_window(
        db, tenant_oid, start=prev_window_start, end=prev_window_end, project_oid=project_oid
    )
    requests = int(current_agg.get("ai_calls", 0))
    prompt_tokens = int(current_agg.get("prompt_tokens", 0))
    completion_tokens = int(current_agg.get("completion_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens

    prev_requests = int(previous_agg.get("ai_calls", 0))
    prev_tokens = int(previous_agg.get("prompt_tokens", 0)) + int(previous_agg.get("completion_tokens", 0))

    def _pct(cur: int, prev: int) -> float | None:
        if prev <= 0:
            return None if cur == 0 else 100.0
        return round(((cur - prev) / prev) * 100, 1)

    # ---- Content KPIs ----
    decisions_total = await db.canvases.count_documents({"tenant_id": tenant_id})
    decisions_window = await db.canvases.count_documents(
        {"tenant_id": tenant_id, "created_at": {"$gte": window_start}}
    )
    project_decisions = await db.canvases.count_documents(
        {"tenant_id": tenant_id, "project_id": project_id}
    )
    prd_docs = await db.documents.count_documents(
        {"tenant_id": tenant_id, "title": {"$regex": r"^PRD", "$options": "i"}}
    )
    prd_runs_window = await db.prd_generation_jobs.count_documents(
        {"updated_at": {"$gte": window_start}}
    )
    active_prd_runs = await db.prd_generation_jobs.count_documents({"status": "processing"})

    # ---- Tasks / sprints (project-scoped) ----
    sprint_filter = {"tenant_id": tenant_oid, "project_id": project_oid, "deleted_at": None}
    sprints_total = await db.sprints.count_documents(sprint_filter)
    workflow_filter = {"tenant_id": tenant_oid, "project_id": project_oid}
    tasks_total = await db.workflow_tasks.count_documents(workflow_filter)
    tasks_by_status = {
        "todo": await db.workflow_tasks.count_documents({**workflow_filter, "status": "todo"}),
        "in_progress": await db.workflow_tasks.count_documents({**workflow_filter, "status": "in_progress"}),
        "done": await db.workflow_tasks.count_documents({**workflow_filter, "status": "done"}),
        "blocked": await db.workflow_tasks.count_documents({**workflow_filter, "status": "blocked"}),
    }

    # ---- Notifications / issues ----
    notify_filter = {"tenant_id": tenant_oid}
    unread_notifications = await db.notifications.count_documents(
        {"tenant_id": tenant_oid, "is_read": {"$ne": True}}
    )
    notifications_total = await db.notifications.count_documents(notify_filter)
    open_issues = await db.feedback.count_documents({"tenant_id": tenant_oid, "status": {"$ne": "resolved"}})
    feedback_total = await db.feedback.count_documents({"tenant_id": tenant_oid})

    # ---- Quotas (progress bars) ----
    quotas = [
        {
            "key": "ai_tokens",
            "title": "AI Tokens",
            "used": total_tokens,
            "total": ai_limit,
            "unit": " tokens",
        },
        {
            "key": "projects",
            "title": "Projects",
            "used": capacity["projects_used"],
            "total": projects_limit,
            "unit": "",
        },
        {
            "key": "team_members",
            "title": "Team Members",
            "used": capacity["team_members_used"],
            "total": members_limit,
            "unit": "",
        },
    ]
    quota_exceeded = any(
        q["total"] is not None and q["used"] >= q["total"] for q in quotas
    )

    # ---- Activity timeline ----
    recent_activity_query = {
        "tenant_id": tenant_oid,
        "$or": [
            {"entity_id": project_id},
            {"metadata.project_id": project_id},
            {"action": {"$regex": "project", "$options": "i"}},
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
            "created_at": _json_safe(doc.get("created_at")),
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
            "created_at": _json_safe(project.get("created_at")),
            "updated_at": _json_safe(project.get("updated_at")),
        },
        "kpis": {
            "members": members_count,
            "members_limit": members_limit,
            "projects_used": capacity["projects_used"],
            "projects_limit": projects_limit,
            "decisions_total": decisions_total,
            "decisions_window": decisions_window,
            "project_decisions": project_decisions,
            "prd_docs": prd_docs,
            "prd_runs_window": prd_runs_window,
            "active_prd_runs": active_prd_runs,
            "sprints_total": sprints_total,
            "tasks_total": tasks_total,
            "tasks_by_status": tasks_by_status,
            "unread_notifications": unread_notifications,
            "notifications_total": notifications_total,
            "open_issues": open_issues,
            "feedback_total": feedback_total,
        },
        "plan": {
            "id": plan_id,
            "name": plan["name"] if isinstance(plan, dict) else str(plan),
            "quota_exceeded": quota_exceeded,
            "quotas": quotas,
        },
        "usage_delta": {
            "requests_pct": _pct(requests, prev_requests),
            "tokens_pct": _pct(total_tokens, prev_tokens),
        },
        "llm_usage": {
            "window_days": days,
            "requests": requests,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": estimate_cost(prompt_tokens, completion_tokens),
            "avg_tokens_per_request": round(total_tokens / requests, 1) if requests else 0.0,
            "max_tokens_per_request": 0,
            "token_budget_per_request": 0,
            "token_headroom_percent": 100.0,
            "daily": await _usage_series(
                db, tenant_oid, start=window_start, end=now + timedelta(days=1), project_oid=project_oid
            ),
        },
        "requirements": {"status": None, "created_at": None, "updated_at": None},
        "prd": {
            "latest_status": None,
            "latest_created_at": None,
            "latest_updated_at": None,
            "latest_completed_at": None,
            "latest_version": None,
        },
        "recent_decisions": [],
        "recent_activity": recent_activity,
    }
