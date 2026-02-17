from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import MongoClient

from app.core.config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _determine_plan(doc: dict) -> str:
    legacy_type = doc.get("type")
    if legacy_type in {"trial", "starter", "team", "enterprise"}:
        return legacy_type
    if legacy_type == "paid":
        return "starter"
    return "starter"


def _determine_status(doc: dict, now: datetime) -> str:
    legacy_status = doc.get("status")
    if legacy_status in {"suspended", "expired", "active"}:
        return legacy_status
    if legacy_status == "inactive":
        return "expired"

    expiry_date = doc.get("expiry_date") or doc.get("expires_at")
    if expiry_date and now > expiry_date:
        return "expired"
    return "active"


def _determine_dates(doc: dict, now: datetime) -> tuple[datetime, datetime]:
    start_date = doc.get("start_date") or doc.get("created_at") or now
    expiry_date = doc.get("expiry_date") or doc.get("expires_at")
    if not expiry_date:
        if doc.get("type") == "trial":
            expiry_date = start_date + timedelta(days=settings.trial_days)
        else:
            # Fallback for paid licenses missing expiry: assume 1 year from start_date.
            expiry_date = start_date + timedelta(days=365)
    return start_date, expiry_date


def _normalize_license(doc: dict, now: datetime) -> dict:
    start_date, expiry_date = _determine_dates(doc, now)
    plan = _determine_plan(doc)
    status = _determine_status(doc, now)
    grace_days = doc.get("grace_period_days", settings.trial_grace_days)
    return {
        "plan": plan,
        "status": status,
        "start_date": start_date,
        "expiry_date": expiry_date,
        "grace_period_days": grace_days,
    }


def _enforce_one_license_per_tenant(collection, apply: bool) -> int:
    now = _utcnow()
    deleted = 0
    pipeline = [
        {"$match": {"$or": [{"deleted_at": {"$exists": False}}, {"deleted_at": None}]}},
        {
            "$group": {
                "_id": "$tenant_id",
                "licenses": {"$push": {"_id": "$_id", "created_at": "$created_at"}},
                "count": {"$sum": 1},
            }
        },
        {"$match": {"count": {"$gt": 1}}},
    ]
    for group in collection.aggregate(pipeline):
        licenses = sorted(
            group["licenses"],
            key=lambda item: item.get("created_at") or now,
            reverse=True,
        )
        keep = licenses[0]["_id"]
        for doc in licenses[1:]:
            if apply:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"deleted_at": now, "deleted_by": None}},
                )
            deleted += 1
        print(f"Tenant {group['_id']} keeping {keep}, soft-deleting {len(licenses) - 1}")
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy licenses to new schema")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    client = MongoClient(settings.mongo_uri)
    db = client[settings.mongo_db]
    licenses = db.licenses

    now = _utcnow()
    updated = 0
    skipped = 0

    cursor = licenses.find({"$or": [{"deleted_at": {"$exists": False}}, {"deleted_at": None}]})
    for doc in cursor:
        if "plan" in doc and "expiry_date" in doc:
            skipped += 1
            continue

        updates = _normalize_license(doc, now)
        updates["deleted_at"] = doc.get("deleted_at", None)
        updates["deleted_by"] = doc.get("deleted_by", None)
        set_doc = {**updates}
        unset_doc = {"type": "", "expires_at": "", "user_id": ""}
        if args.apply:
            licenses.update_one(
                {"_id": doc["_id"]},
                {"$set": set_doc, "$unset": unset_doc},
            )
        updated += 1

    deleted = _enforce_one_license_per_tenant(licenses, args.apply)

    print(f"Dry run: {not args.apply}")
    print(f"Updated licenses: {updated}")
    print(f"Skipped licenses: {skipped}")
    print(f"Soft-deleted duplicates: {deleted}")


if __name__ == "__main__":
    main()
