import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bson import ObjectId

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.mongo import get_db
from app.utils.security import hash_password


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


async def _resolve_or_create_tenant(tenant_id: str | None, tenant_name: str | None) -> ObjectId:
    db = get_db()
    if tenant_id:
        oid = _oid(tenant_id)
        tenant = await db.tenants.find_one({"_id": oid})
        if not tenant:
            raise ValueError("Tenant not found for provided --tenant-id")
        return oid

    if not tenant_name:
        raise ValueError("Provide --tenant-name when --tenant-id is not set")

    slug = _slugify(tenant_name)
    existing = await db.tenants.find_one({"slug": slug})
    if existing:
        return existing["_id"]

    base_slug = slug
    suffix = 1
    while await db.tenants.find_one({"slug": slug}):
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    now = _utcnow()
    result = await db.tenants.insert_one({"name": tenant_name, "slug": slug, "created_at": now})
    return result.inserted_id


async def _upsert_user(tenant_oid: ObjectId, email: str, password: str, role: str) -> ObjectId:
    db = get_db()
    normalized_email = email.lower().strip()
    user = await db.users.find_one({"tenant_id": tenant_oid, "email": normalized_email})
    if user:
        await db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "password_hash": hash_password(password),
                    "role": role,
                    "provider": "password",
                    "updated_at": _utcnow(),
                }
            },
        )
        return user["_id"]

    now = _utcnow()
    result = await db.users.insert_one(
        {
            "tenant_id": tenant_oid,
            "email": normalized_email,
            "password_hash": hash_password(password),
            "role": role,
            "provider": "password",
            "created_at": now,
            "last_login_at": None,
        }
    )
    return result.inserted_id


async def _set_enterprise_license(tenant_oid: ObjectId) -> ObjectId:
    db = get_db()
    now = _utcnow()
    await db.licenses.update_many(
        {
            "tenant_id": {"$in": [tenant_oid, str(tenant_oid)]},
            "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
        },
        {"$set": {"deleted_at": now, "deleted_by": None}},
    )
    result = await db.licenses.insert_one(
        {
            "tenant_id": tenant_oid,
            "plan": "enterprise",
            "status": "active",
            "start_date": now,
            "expiry_date": now + timedelta(days=3650),
            "grace_period_days": 30,
            "created_at": now,
            "deleted_at": None,
            "deleted_by": None,
        }
    )
    return result.inserted_id


async def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update user and assign enterprise license")
    parser.add_argument("--email", required=True, help="User email")
    parser.add_argument("--password", required=True, help="User password")
    parser.add_argument("--role", default="owner", help="owner | admin | member")
    parser.add_argument("--tenant-id", default=None, help="Existing tenant ObjectId")
    parser.add_argument("--tenant-name", default=None, help="New/existing tenant name")
    args = parser.parse_args()

    tenant_oid = await _resolve_or_create_tenant(args.tenant_id, args.tenant_name)
    user_oid = await _upsert_user(tenant_oid, args.email, args.password, args.role)
    license_oid = await _set_enterprise_license(tenant_oid)

    print("created_or_updated_user")
    print(f"tenant_id={tenant_oid}")
    print(f"user_id={user_oid}")
    print(f"license_id={license_oid}")
    print("plan=enterprise status=active")


if __name__ == "__main__":
    asyncio.run(main())

