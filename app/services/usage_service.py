"""Usage tracking service.

Records AI/token consumption at the call site and serves real-time usage
overviews for the Usage page. Cycle totals live in ``licenses.usage`` (the same
counters used by quota enforcement); per-day + optional per-project rows live in
``usage_daily`` so the UI can render charts and period slices.

Recording is best-effort: it must never break the request that consumed the
tokens, so failures are logged and swallowed.
"""

from __future__ import annotations

import logging

from bson import ObjectId

from app.core.plans import get_plan, plan_quota
from app.db.mongo import get_db
from app.services.license_service import compute_usage, get_or_create_license

logger = logging.getLogger("decisionvault.usage")


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate (``~4 chars/token``) used when upstream usage is absent.

    Good enough for capacity display, not for billing.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


async def record_usage(
    tenant_id: str,
    *,
    project_id: str | None = None,
    feature: str = "generic",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record token consumption for a tenant, best-effort.

    Increments the cycle counters in ``licenses.usage`` and the daily row in
    ``usage_daily``. Never raises.
    """
    prompt_tokens = max(0, int(prompt_tokens or 0))
    completion_tokens = max(0, int(completion_tokens or 0))
    if prompt_tokens == 0 and completion_tokens == 0:
        return

    try:
        db = get_db()
        oid = _oid(tenant_id)
        now = _utcnow()

        # Ensure a license doc exists, then increment its cycle counters.
        await get_or_create_license(tenant_id)
        await db.licenses.update_one(
            {"tenant_id": oid},
            {
                "$inc": {
                    "usage.prompt_tokens": prompt_tokens,
                    "usage.completion_tokens": completion_tokens,
                    "usage.ai_calls": 1,
                },
                "$set": {"updated_at": now},
            },
        )

        # Daily row for charts / period slices (org-level or per-project).
        date_key = now.strftime("%Y-%m-%d")
        daily_filter: dict = {"tenant_id": oid, "date": date_key}
        if project_id:
            daily_filter["project_id"] = _oid(project_id)
        await db.usage_daily.update_one(
            daily_filter,
            {
                "$inc": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "ai_calls": 1,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
    except Exception:
        logger.exception(
            "usage_record_failed tenant_id=%s feature=%s", tenant_id, feature
        )


def _shift_month(d) :
    """Shift a first-of-month datetime back by one month."""
    if d.month == 1:
        return d.replace(year=d.year - 1, month=12, day=1)
    return d.replace(month=d.month - 1, day=1)


def _fmt(n: int | float | None) -> str:
    if n is None:
        return "Unlimited"
    return f"{int(round(n)):,}"


async def _aggregate_period(
    tenant_id: str, *, start, end, project_id: str | None = None
) -> dict:
    db = get_db()
    match: dict = {
        "tenant_id": _oid(tenant_id),
        "date": {"$gte": start.strftime("%Y-%m-%d"), "$lt": end.strftime("%Y-%m-%d")},
    }
    if project_id:
        match["project_id"] = _oid(project_id)
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


async def get_usage_overview(
    tenant_id: str,
    *,
    period: str = "current",
    project_id: str | None = None,
) -> dict:
    """Build the payload consumed by the Usage page (see usage-tracking spec §7.1)."""
    license_doc = await get_or_create_license(tenant_id)
    plan_id = license_doc["plan"]
    plan = get_plan(plan_id)

    period_start = license_doc.get("current_period_start")
    period_end = license_doc.get("current_period_end")
    if period == "previous" and period_start:
        period_start, period_end = _shift_month(period_start), period_start
    if not period_start or not period_end:
        now = _utcnow()
        period_start = period_start or now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = period_end or now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    totals = await _aggregate_period(
        tenant_id, start=period_start, end=period_end, project_id=project_id
    )
    prompt_tokens = int(totals.get("prompt_tokens", 0))
    completion_tokens = int(totals.get("completion_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens
    ai_calls = int(totals.get("ai_calls", 0))

    # Live plan capacity (org-level counts; project filter only affects tokens).
    capacity = await compute_usage(tenant_id, plan_id)

    ai_limit = plan_quota(plan_id, "ai_tokens")
    projects_limit = plan_quota(plan_id, "projects")
    members_limit = plan_quota(plan_id, "team_members")
    storage_limit = plan_quota(plan_id, "storage_mb")

    quotas = [
        ("ai_tokens", total_tokens, ai_limit),
        ("projects", capacity["projects_used"], projects_limit),
        ("team_members", capacity["team_members_used"], members_limit),
        ("storage_mb", capacity["storage_mb_used"], storage_limit),
    ]
    quota_exceeded = any(
        limit is not None and used >= limit for _, used, limit in quotas
    )

    summary = [
        {
            "key": "ai_tokens",
            "title": "AI Tokens",
            "used": total_tokens,
            "total": ai_limit,
            "unit": " tokens",
            "premium": False,
        },
        {
            "key": "ai_calls",
            "title": "AI Calls",
            "used": ai_calls,
            "total": None,
            "unit": "",
            "premium": False,
        },
        {
            "key": "projects",
            "title": "Projects",
            "used": capacity["projects_used"],
            "total": projects_limit,
            "unit": "",
            "premium": False,
        },
        {
            "key": "team_members",
            "title": "Team Members",
            "used": capacity["team_members_used"],
            "total": members_limit,
            "unit": "",
            "premium": False,
        },
        {
            "key": "storage_mb",
            "title": "Storage",
            "used": round(capacity["storage_mb_used"], 2),
            "total": storage_limit,
            "unit": " MB",
            "premium": False,
        },
    ]

    overage = max(0, total_tokens - ai_limit) if ai_limit is not None else 0
    detail_rows = [
        {
            "label": f"Included in {plan['name'] if plan else 'Free'} plan",
            "value": _fmt(ai_limit) + " tokens" if ai_limit is not None else "Unlimited",
        },
        {"label": "Used in period", "value": f"{_fmt(total_tokens)} tokens"},
        {"label": "Overage in period", "value": f"{_fmt(overage)} tokens", "highlight": True},
    ]
    capacity_rows = [
        {"label": "Projects", "value": f"{_fmt(capacity['projects_used'])} / {_fmt(projects_limit)}"},
        {"label": "Team Members", "value": f"{_fmt(capacity['team_members_used'])} / {_fmt(members_limit)}"},
        {"label": "Storage", "value": f"{_fmt(capacity['storage_mb_used'])} MB / {_fmt(storage_limit)} MB"},
    ]

    db = get_db()
    daily_match: dict = {
        "tenant_id": _oid(tenant_id),
        "date": {"$gte": period_start.strftime("%Y-%m-%d"), "$lt": period_end.strftime("%Y-%m-%d")},
    }
    if project_id:
        daily_match["project_id"] = _oid(project_id)
    daily_cursor = db.usage_daily.find(daily_match).sort("date", 1)
    daily = [
        {
            "date": doc["date"],
            "total_tokens": int(doc.get("prompt_tokens", 0) + doc.get("completion_tokens", 0)),
            "ai_calls": int(doc.get("ai_calls", 0)),
        }
        async for doc in daily_cursor
    ]

    return {
        "period": {
            "key": "current" if period != "previous" else "previous",
            "start": period_start,
            "end": period_end,
        },
        "plan": {
            "id": plan_id,
            "name": plan["name"] if plan else plan_id.title(),
            "quota_exceeded": quota_exceeded,
        },
        "summary": summary,
        "detail": {
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "ai_calls": ai_calls,
            },
            "rows": detail_rows,
            "capacity": capacity_rows,
        },
        "daily": daily,
    }
