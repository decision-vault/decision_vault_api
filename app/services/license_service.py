from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.db.mongo import get_db
from app.utils.serialize import serialize_doc


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if not value:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _normalize_license(doc: dict) -> dict:
    return serialize_doc(doc)


async def get_current_license(tenant_id: str) -> dict | None:
    db = get_db()
    tenant_match = {"$in": [_oid(tenant_id), tenant_id]}
    license_doc = await db.licenses.find_one(
        {
            "tenant_id": tenant_match,
            "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
        }
    )
    if not license_doc:
        return None
    return _normalize_license(license_doc)


def evaluate_license_status(license_doc: dict, now: datetime | None = None) -> dict:
    now = _ensure_aware(now or _utcnow())
    status = license_doc.get("status", "active")
    if status == "trial":
        return {"status": "active", "read_only": False}
    if status == "grace":
        return {"status": "grace", "read_only": True}
    if status == "suspended":
        return {"status": "suspended", "read_only": True}
    if status == "expired":
        return {"status": "expired", "read_only": True}

    expiry_date = _ensure_aware(license_doc.get("expiry_date"))
    grace_days = license_doc.get("grace_period_days", 7)
    if expiry_date and now > expiry_date + timedelta(days=grace_days):
        return {"status": "suspended", "read_only": True}
    if expiry_date and now > expiry_date:
        return {"status": "expired", "read_only": True}
    return {"status": "active", "read_only": False}


def license_status_banner(license_doc: dict, now: datetime | None = None) -> dict:
    now = _ensure_aware(now or _utcnow())
    expiry_date = _ensure_aware(license_doc.get("expiry_date"))
    grace_days = license_doc.get("grace_period_days", 7)
    status_info = evaluate_license_status(license_doc, now)
    days_remaining = None
    grace_remaining = None
    grace_end_date = None
    if expiry_date:
        delta = (expiry_date - now).days
        days_remaining = delta
        grace_end_date = expiry_date + timedelta(days=grace_days)
        grace_remaining = (grace_end_date - now).days

    message = None
    if status_info["status"] == "active" and days_remaining is not None and days_remaining <= 7:
        message = f"Trial expires in {max(days_remaining, 0)} days."
    if status_info["status"] in {"expired", "grace"}:
        message = "Your license has expired. You are in read-only mode."
    if status_info["status"] == "suspended":
        message = "Your license is suspended. Please contact support."

    return {
        "status": status_info["status"],
        "read_only": status_info["read_only"],
        "days_remaining": days_remaining,
        "grace_remaining": grace_remaining,
        "grace_end_date": grace_end_date,
        "message": message,
    }


async def create_license(tenant_id: str, user_id: str, payload: dict) -> dict:
    db = get_db()
    _ = user_id
    existing = await db.licenses.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
        }
    )
    if existing:
        raise ValueError("License already exists for tenant")
    doc = {
        "tenant_id": _oid(tenant_id),
        "plan": payload["plan"],
        "status": payload.get("status", "active"),
        "start_date": payload["start_date"],
        "expiry_date": payload["expiry_date"],
        "grace_period_days": payload.get("grace_period_days", 7),
        "created_at": _utcnow(),
        "deleted_at": None,
    }
    result = await db.licenses.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _normalize_license(doc)


async def update_license(tenant_id: str, license_id: str, updates: dict) -> dict | None:
    db = get_db()
    update_fields = {k: v for k, v in updates.items() if v is not None}
    if not update_fields:
        return None
    doc = await db.licenses.find_one_and_update(
        {
            "_id": _oid(license_id),
            "tenant_id": _oid(tenant_id),
            "deleted_at": None,
        },
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        return None
    return _normalize_license(doc)


async def delete_license(tenant_id: str, license_id: str, deleted_by: str) -> bool:
    db = get_db()
    result = await db.licenses.update_one(
        {
            "_id": _oid(license_id),
            "tenant_id": _oid(tenant_id),
            "deleted_at": None,
        },
        {"$set": {"deleted_at": _utcnow(), "deleted_by": _oid(deleted_by)}},
    )
    return result.modified_count == 1


async def restore_license(tenant_id: str, license_id: str) -> dict | None:
    db = get_db()
    doc = await db.licenses.find_one_and_update(
        {"_id": _oid(license_id), "tenant_id": _oid(tenant_id)},
        {"$set": {"deleted_at": None, "deleted_by": None}},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        return None
    return _normalize_license(doc)
