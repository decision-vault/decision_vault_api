import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bson import ObjectId

# Allow running the script directly from /scripts without `-m`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.mongo import get_db


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_oid(value: str) -> ObjectId:
    return ObjectId(value)


async def _target_tenants(tenant_id: str | None, all_tenants: bool) -> list[ObjectId]:
    db = get_db()
    if all_tenants:
        return [doc["_id"] async for doc in db.tenants.find({}, {"_id": 1})]
    if not tenant_id:
        raise ValueError("Provide --tenant-id or use --all-tenants")
    return [_to_oid(tenant_id)]


async def _upsert_enterprise_license(tenant_oid: ObjectId) -> None:
    db = get_db()
    now = _utcnow()
    expiry = now + timedelta(days=3650)

    await db.licenses.update_many(
        {
            "tenant_id": tenant_oid,
            "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
        },
        {"$set": {"deleted_at": now, "deleted_by": None}},
    )

    await db.licenses.insert_one(
        {
            "tenant_id": tenant_oid,
            "plan": "enterprise",
            "status": "active",
            "start_date": now,
            "expiry_date": expiry,
            "grace_period_days": 30,
            "created_at": now,
            "deleted_at": None,
            "deleted_by": None,
        }
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed enterprise license with full access")
    parser.add_argument("--tenant-id", type=str, default=None, help="Tenant ObjectId")
    parser.add_argument(
        "--all-tenants",
        action="store_true",
        help="Seed enterprise license for all tenants",
    )
    args = parser.parse_args()

    tenants = await _target_tenants(args.tenant_id, args.all_tenants)
    for tenant_oid in tenants:
        await _upsert_enterprise_license(tenant_oid)
        print(f"seeded enterprise license for tenant={tenant_oid}")


if __name__ == "__main__":
    asyncio.run(main())
