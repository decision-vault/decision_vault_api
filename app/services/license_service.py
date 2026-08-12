"""Licensing & usage service.

One ``licenses`` document per tenant holds the active plan, billing details,
invoices, payment methods, credit balance and usage counters. Quota checks are
computed live against the database so they always reflect reality.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.core.config import settings
from app.core.plans import (
    DEFAULT_SIGNUP_PLAN,
    PLAN_FEATURES,
    get_plan,
    normalize_plan,
    plan_catalog,
    plan_has_feature,
    plan_quota,
)
from app.db.mongo import get_db
from app.utils.serialize import serialize_doc


class QuotaExceededError(Exception):
    def __init__(self, quota_key: str, limit: int | None, used: int, plan: str):
        super().__init__(
            f"Plan limit reached for {quota_key} ({used}/{limit or 'unlimited'})"
        )
        self.quota_key = quota_key
        self.limit = limit
        self.used = used
        self.plan = plan


class PlanError(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _month_period(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def _build_license(
    tenant_id: ObjectId,
    *,
    plan: str = DEFAULT_SIGNUP_PLAN,
    billing_cycle: str = "monthly",
    billing_email: str | None = None,
) -> dict:
    now = _utcnow()
    period_start, period_end = _month_period(now)
    return {
        "tenant_id": tenant_id,
        "plan": plan,
        "status": "active",
        "billing_cycle": billing_cycle,
        "start_date": now,
        "current_period_start": period_start,
        "current_period_end": period_end,
        "grace_period_days": settings.trial_grace_days,
        "billing_email": billing_email,
        "additional_emails": [],
        "address": {
            "full_name": None,
            "country": None,
            "address": None,
            "tax_id": None,
            "tax_id_type": None,
        },
        "spend_cap_enabled": True,
        "credit_balance": 0.0,
        "invoices": [],
        "invoice_seq": 0,
        "payment_methods": [],
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "usage": {},
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "deleted_by": None,
    }


async def get_license(tenant_id: str) -> dict | None:
    db = get_db()
    doc = await db.licenses.find_one({"tenant_id": _oid(tenant_id)})
    if not doc:
        return None
    doc = serialize_doc(doc)
    doc["plan"] = normalize_plan(doc.get("plan"))
    return doc


async def get_or_create_license(tenant_id: str) -> dict:
    db = get_db()
    doc = await db.licenses.find_one({"tenant_id": _oid(tenant_id)})
    if doc:
        plan = normalize_plan(doc.get("plan"))
        if plan != doc.get("plan"):
            await db.licenses.update_one(
                {"_id": doc["_id"]}, {"$set": {"plan": plan, "updated_at": _utcnow()}}
            )
            doc["plan"] = plan
        return serialize_doc(doc)
    doc = _build_license(_oid(tenant_id), plan=DEFAULT_SIGNUP_PLAN)
    result = await db.licenses.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_doc(doc)


async def create_license(
    tenant_id: str,
    *,
    plan: str = DEFAULT_SIGNUP_PLAN,
    billing_cycle: str = "monthly",
    billing_email: str | None = None,
) -> dict:
    plan = normalize_plan(plan)
    if plan not in ("free", "lite", "pro"):
        raise PlanError("Invalid plan")
    doc = _build_license(
        _oid(tenant_id), plan=plan, billing_cycle=billing_cycle, billing_email=billing_email
    )
    result = await get_db().licenses.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_doc(doc)


def _count_used(usage: dict, quota_key: str) -> int:
    return int(usage.get(f"{quota_key}_used", 0))


async def compute_usage(tenant_id: str, plan: str) -> dict:
    """Compute live usage for the tenant against the current plan."""
    db = get_db()
    oid = _oid(tenant_id)
    now = _utcnow()

    projects_used = await db.projects.count_documents(
        {"tenant_id": oid, "deleted_at": None}
    )
    members_used = await db.users.count_documents(
        {
            "tenant_id": oid,
            "deleted_at": None,
            "$or": [{"is_active": True}, {"is_active": {"$exists": False}}],
        }
    )
    # Storage is tracked via a cumulative counter where available.
    license_doc = await db.licenses.find_one({"tenant_id": oid})
    usage = (license_doc or {}).get("usage") or {}
    storage_used = round(_count_used(usage, "storage_mb"), 2)
    ai_tokens_used = int(
        _count_used(usage, "prompt_tokens") + _count_used(usage, "completion_tokens")
    )

    return {
        "projects_used": projects_used,
        "team_members_used": members_used,
        "storage_mb_used": storage_used,
        "ai_tokens_used": ai_tokens_used,
        "period_start": (license_doc or {}).get("current_period_start") or now,
        "period_end": (license_doc or {}).get("current_period_end") or now,
    }


def _decorate_invoices(invoices: list[dict]) -> list[dict]:
    decorated = sorted(
        invoices or [], key=lambda inv: inv.get("created_at") or _utcnow(), reverse=True
    )
    for inv in decorated:
        inv["id"] = inv.get("number")
        inv["amount"] = float(inv.get("amount", 0))
        inv["currency"] = inv.get("currency", "USD")
        inv["status"] = inv.get("status", "paid")
    return decorated


async def get_billing_overview(tenant_id: str) -> dict:
    license_doc = await get_or_create_license(tenant_id)
    plan_id = license_doc["plan"]
    plan = get_plan(plan_id)
    usage = await compute_usage(tenant_id, plan_id)

    quotas: dict[str, dict] = {}
    for key in ("projects", "team_members", "storage_mb", "ai_tokens"):
        limit = plan_quota(plan_id, key)
        used = usage[f"{key}_used"]
        quotas[key] = {"limit": limit, "used": used, "remaining": None if limit is None else max(0, limit - used)}

    license_payload = {
        "tenant_id": license_doc["tenant_id"],
        "plan": plan_id,
        "status": license_doc.get("status", "active"),
        "billing_cycle": license_doc.get("billing_cycle", "monthly"),
        "start_date": license_doc.get("start_date"),
        "current_period_start": license_doc.get("current_period_start"),
        "current_period_end": license_doc.get("current_period_end"),
        "grace_period_days": license_doc.get("grace_period_days", 0),
        "features": PLAN_FEATURES.get(plan_id, []) if plan else [],
        "quotas": quotas,
        "usage": usage,
    }

    return {
        "plan": license_payload,
        "plan_name": plan["name"] if plan else plan_id.title(),
        "price": float(plan["price_monthly"] if plan else 0),
        "currency": plan["currency"] if plan else "USD",
        "spend_cap_enabled": bool(license_doc.get("spend_cap_enabled", True)),
        "credit_balance": float(license_doc.get("credit_balance", 0) or 0),
        "billing_email": license_doc.get("billing_email"),
        "additional_emails": license_doc.get("additional_emails") or [],
        "address": license_doc.get("address") or {},
        "invoices": _decorate_invoices(license_doc.get("invoices") or []),
        "payment_methods": license_doc.get("payment_methods") or [],
        "stripe_enabled": bool(settings.stripe_secret_key),
        "stripe_customer_id": license_doc.get("stripe_customer_id"),
    }


def _make_invoice(license_doc: dict, *, plan: str, amount: float, status: str = "paid") -> dict:
    seq = int(license_doc.get("invoice_seq", 0)) + 1
    now = _utcnow()
    return {
        "number": f"INV-{seq:06d}",
        "plan": plan,
        "amount": amount,
        "currency": get_plan(plan)["currency"] if get_plan(plan) else "USD",
        "status": status,
        "created_at": now,
        "period_start": license_doc.get("current_period_start"),
        "period_end": license_doc.get("current_period_end"),
        "download_url": None,
    }


async def change_plan(
    tenant_id: str, plan: str, billing_cycle: str = "monthly"
) -> dict:
    plan = normalize_plan(plan)
    if plan not in ("free", "lite", "pro"):
        raise PlanError("Invalid plan")

    db = get_db()
    license_doc = await get_or_create_license(tenant_id)
    oid = _oid(tenant_id)
    now = _utcnow()

    previous = license_doc.get("plan")
    price = get_plan(plan)["price_yearly"] if billing_cycle == "yearly" else get_plan(plan)["price_monthly"]

    updates = {
        "plan": plan,
        "billing_cycle": billing_cycle,
        "updated_at": now,
    }
    if previous != plan or plan != "free":
        updates["current_period_start"], updates["current_period_end"] = _month_period(now)

    # In local mode (no Stripe keys) a plan change to a paid plan creates an invoice.
    if not settings.stripe_secret_key and price > 0:
        invoice = _make_invoice(license_doc, plan=plan, amount=float(price))
        updates["invoice_seq"] = int(license_doc.get("invoice_seq", 0)) + 1
        updates["invoices"] = list(license_doc.get("invoices") or []) + [invoice]

    await db.licenses.update_one({"tenant_id": _oid(tenant_id)}, {"$set": updates})

    from app.services.audit_service import log_event

    await log_event(
        tenant_id=tenant_id,
        actor_id=None,
        action="billing.plan.changed",
        entity_type="license",
        entity_id=tenant_id,
        metadata={"from": previous, "to": plan, "billing_cycle": billing_cycle},
    )
    return await get_billing_overview(tenant_id)


async def update_billing_details(tenant_id: str, payload: dict) -> dict:
    db = get_db()
    updates: dict = {}
    if "billing_email" in payload:
        updates["billing_email"] = payload["billing_email"]
    if "additional_emails" in payload:
        updates["additional_emails"] = payload["additional_emails"] or []
    if "spend_cap_enabled" in payload:
        updates["spend_cap_enabled"] = bool(payload["spend_cap_enabled"])
    if "address" in payload and payload["address"] is not None:
        merged = dict(payload["address"])
        updates["address"] = merged
    updates["updated_at"] = _utcnow()
    if updates:
        await db.licenses.update_one({"tenant_id": _oid(tenant_id)}, {"$set": updates})
    return await get_billing_overview(tenant_id)


async def list_invoices(tenant_id: str) -> list[dict]:
    license_doc = await get_or_create_license(tenant_id)
    return _decorate_invoices(license_doc.get("invoices") or [])


async def add_payment_method(tenant_id: str, token: str | None = None) -> dict:
    db = get_db()
    license_doc = await get_or_create_license(tenant_id)
    existing = license_doc.get("payment_methods") or []
    if existing:
        return await get_billing_overview(tenant_id)
    method = {
        "id": str(ObjectId()),
        "brand": "Visa",
        "last4": "4242",
        "exp_month": 12,
        "exp_year": 2027,
        "is_default": True,
    }
    await db.licenses.update_one(
        {"tenant_id": _oid(tenant_id)},
        {"$set": {"payment_methods": existing + [method], "updated_at": _utcnow()}},
    )
    return await get_billing_overview(tenant_id)


async def remove_payment_method(tenant_id: str, method_id: str) -> dict:
    db = get_db()
    license_doc = await get_or_create_license(tenant_id)
    methods = [m for m in (license_doc.get("payment_methods") or []) if m.get("id") != method_id]
    await db.licenses.update_one(
        {"tenant_id": _oid(tenant_id)},
        {"$set": {"payment_methods": methods, "updated_at": _utcnow()}},
    )
    return await get_billing_overview(tenant_id)


async def redeem_credit(tenant_id: str, code: str) -> dict:
    db = get_db()
    normalized = code.strip().upper()
    if not normalized:
        raise PlanError("Invalid code")
    amounts = {"DV10": 10.0, "DV25": 25.0, "DV50": 50.0}
    if normalized not in amounts:
        raise PlanError("Invalid credit code")
    license_doc = await get_or_create_license(tenant_id)
    new_balance = float(license_doc.get("credit_balance", 0) or 0) + amounts[normalized]
    await db.licenses.update_one(
        {"tenant_id": _oid(tenant_id)},
        {"$set": {"credit_balance": new_balance, "updated_at": _utcnow()}},
    )
    return await get_billing_overview(tenant_id)


async def increment_usage(tenant_id: str, key: str, amount: int = 1) -> None:
    db = get_db()
    await db.licenses.update_one(
        {"tenant_id": _oid(tenant_id)},
        {
            "$inc": {f"usage.{key}": amount},
            "$set": {"updated_at": _utcnow()},
            "$setOnInsert": {"created_at": _utcnow()},
        },
        upsert=True,
    )


async def enforce_quota(tenant_id: str, quota_key: str) -> None:
    """Raise ``QuotaExceededError`` if the tenant has used its plan limit."""
    license_doc = await get_or_create_license(tenant_id)
    plan = license_doc["plan"]
    limit = plan_quota(plan, quota_key)
    if limit is None:
        return
    usage = await compute_usage(tenant_id, plan)
    used = usage[f"{quota_key}_used"]
    if used >= limit:
        raise QuotaExceededError(quota_key, limit, used, plan)


async def enforce_team_member_quota(tenant_id: str, pending_email: str | None = None) -> None:
    """Raise ``QuotaExceededError`` if adding the invitee exceeds the member cap.

    Counts current active members plus pending invites (from both org and
    project invites) for emails that are not already org members.
    """
    license_doc = await get_or_create_license(tenant_id)
    plan = license_doc["plan"]
    limit = plan_quota(plan, "team_members")
    if limit is None:
        return

    db = get_db()
    oid = _oid(tenant_id)
    used = await db.users.count_documents(
        {
            "tenant_id": oid,
            "deleted_at": None,
            "$or": [{"is_active": True}, {"is_active": {"$exists": False}}],
        }
    )

    now = _utcnow()
    pending_emails: set[str] = set()
    for collection_name in ("project_invites", "org_invites"):
        cursor = db[collection_name].find(
            {
                "tenant_id": oid,
                "accepted_at": None,
                "declined_at": None,
                "revoked_at": None,
                "expires_at": {"$gt": now},
            },
            {"email": 1},
        )
        async for doc in cursor:
            if doc.get("email"):
                pending_emails.add(str(doc["email"]).lower())
    if pending_email:
        pending_emails.add(str(pending_email).lower())

    if not pending_emails:
        return

    existing_emails = set()
    cursor = db.users.find(
        {
            "tenant_id": oid,
            "deleted_at": None,
            "email": {"$in": list(pending_emails)},
        },
        {"email": 1},
    )
    async for doc in cursor:
        existing_emails.add(str(doc["email"]).lower())

    would_add = len(pending_emails - existing_emails)
    if used + would_add > limit:
        raise QuotaExceededError("team_members", limit, used + would_add, plan)


async def feature_enabled(tenant_id: str, feature: str) -> bool:
    license_doc = await get_or_create_license(tenant_id)
    return plan_has_feature(license_doc["plan"], feature)


def plans_catalog() -> list[dict]:
    return plan_catalog()
