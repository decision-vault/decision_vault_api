import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.mongo import get_db


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _resolve_owner_oid(db, tenant_oid):
    """Prefer org owner, then admin, then the earliest-created active user."""
    for role in ("owner", "admin"):
        user = await db.users.find_one(
            {"tenant_id": tenant_oid, "role": role, "deleted_at": None},
            sort=[("created_at", 1)],
        )
        if user:
            return user["_id"]
    user = await db.users.find_one(
        {"tenant_id": tenant_oid, "deleted_at": None},
        sort=[("created_at", 1)],
    )
    return user["_id"] if user else None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill projects.owner_id for legacy projects")
    parser.add_argument("--dry-run", action="store_true", help="Only report, do not write")
    parser.add_argument("--project-id", default=None, help="Backfill a single project only")
    args = parser.parse_args()

    db = get_db()
    query = {"deleted_at": None}
    if args.project_id:
        query["_id"] = __import__("bson").ObjectId(args.project_id)

    cursor = db.projects.find(query)
    updated = skipped = 0
    async for project in cursor:
        tenant_oid = project.get("tenant_id")
        if not tenant_oid or project.get("owner_id"):
            skipped += 1
            continue
        owner_oid = await _resolve_owner_oid(db, tenant_oid)
        if not owner_oid:
            skipped += 1
            print(f"[skip] project={project.get('_id')} no org owner found")
            continue
        if args.dry_run:
            print(f"[dry-run] project={project.get('_id')} owner -> {owner_oid}")
            updated += 1
            continue
        await db.projects.update_one(
            {"_id": project["_id"]},
            {"$set": {"owner_id": owner_oid, "updated_at": _utcnow()}},
        )
        print(f"[ok] project={project.get('_id')} owner -> {owner_oid}")
        updated += 1

    mode = "dry-run" if args.dry_run else "write"
    print(f"done ({mode}): updated={updated} skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
