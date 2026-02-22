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


async def get_latest_prd_version(project_id: str) -> dict | None:
    db = get_db()
    doc = await db.prd_versions.find_one({"project_id": project_id}, sort=[("version_number", -1)])
    if not doc:
        return None
    return {
        "project_id": doc.get("project_id"),
        "version_number": doc.get("version_number"),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "markdown_content": doc.get("markdown_content"),
    }


async def list_prd_versions(project_id: str) -> list[dict]:
    db = get_db()
    cursor = db.prd_versions.find({"project_id": project_id}).sort("version_number", -1)
    versions: list[dict] = []
    async for doc in cursor:
        versions.append(
            {
                "project_id": doc.get("project_id"),
                "version_number": doc.get("version_number"),
                "created_by": doc.get("created_by"),
                "created_at": doc.get("created_at"),
            }
        )
    return versions


async def get_prd_version(project_id: str, version_number: int) -> dict | None:
    db = get_db()
    doc = await db.prd_versions.find_one({"project_id": project_id, "version_number": version_number})
    if not doc:
        return None
    return {
        "project_id": doc.get("project_id"),
        "version_number": doc.get("version_number"),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "markdown_content": doc.get("markdown_content"),
    }
