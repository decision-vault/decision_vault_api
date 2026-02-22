from __future__ import annotations

from datetime import datetime, timezone
from app.db.mongo import get_db


async def ensure_prd_table() -> None:
    db = get_db()
    await db.prd_versions.create_index([("project_id", 1), ("version_number", 1)], unique=True)
    await db.prd_versions.create_index([("project_id", 1), ("created_at", -1)])


async def store_prd_version(project_id: str, created_by: str, markdown_content: str) -> dict:
    db = get_db()
    latest = await db.prd_versions.find_one({"project_id": project_id}, sort=[("version_number", -1)])
    next_version = int(latest.get("version_number", 0)) + 1 if latest else 1
    created_at = datetime.now(timezone.utc)
    await db.prd_versions.insert_one(
        {
            "project_id": project_id,
            "version_number": next_version,
            "created_by": created_by,
            "created_at": created_at,
            "markdown_content": markdown_content,
        }
    )
    return {
        "project_id": project_id,
        "version_number": next_version,
        "created_by": created_by,
        "created_at": created_at,
    }
