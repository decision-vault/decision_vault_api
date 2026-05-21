from __future__ import annotations

from datetime import datetime, timezone
import re
from app.db.mongo import get_db


def _looks_like_prd_markdown(markdown: str) -> bool:
    """
    Guardrail: we only want to store finalized PRD markdown in `prd_versions`.
    If a caller accidentally passes the user's raw intake text or an LLM prompt,
    we should refuse to persist it as a PRD version.
    """
    text = (markdown or "").strip()
    if not text:
        return False
    # Common failure modes: raw prompt, raw intake, or JSON blobs.
    if text.startswith("{") and text.endswith("}"):
        return False
    if "Rules:" in text and "Allowed keys:" in text and "Input:" in text:
        return False

    # Keep this aligned with prd_multistep_service's renderer headings.
    # Be tolerant to minor heading format changes (numbered vs unnumbered, casing).
    required_patterns = [
        r"(?m)^#\s*(?:1\.\s*)?Product Requirements Document\s*\(PRD\)\s*$",
        r"(?mi)^##\s*(?:2\.\s*)?Introduction\s*&\s*background\s*$",
        r"(?mi)^##\s*(?:7\.\s*)?non[- ]functional requirements\s*$",
    ]
    return all(re.search(pat, text) for pat in required_patterns)


async def ensure_prd_table() -> None:
    db = get_db()
    await db.prd_versions.create_index([("project_id", 1), ("version_number", 1)], unique=True)
    await db.prd_versions.create_index([("project_id", 1), ("created_at", -1)])


async def store_prd_version(project_id: str, created_by: str, markdown_content: str) -> dict:
    if not _looks_like_prd_markdown(markdown_content):
        raise ValueError("Refusing to store non-PRD content as a PRD version.")
    db = get_db()
    latest = await db.prd_versions.find_one({"project_id": project_id}, sort=[("version_number", -1)])
    next_version = int(latest.get("version_number", 0)) + 1 if latest else 1
    created_at = datetime.now(timezone.utc)
    res = await db.prd_versions.insert_one(
        {
            "project_id": project_id,
            "version_number": next_version,
            "created_by": created_by,
            "created_at": created_at,
            "markdown_content": markdown_content,
        }
    )
    return {
        "doc_id": str(res.inserted_id),
        "project_id": project_id,
        "version_number": next_version,
        "created_by": created_by,
        "created_at": created_at,
    }


async def get_latest_prd_version(project_id: str) -> dict | None:
    db = get_db()
    cursor = db.prd_versions.find({"project_id": project_id}).sort("version_number", -1)
    async for doc in cursor:
        if not _looks_like_prd_markdown(str(doc.get("markdown_content") or "")):
            continue
        return {
            "project_id": doc.get("project_id"),
            "version_number": doc.get("version_number"),
            "created_by": doc.get("created_by"),
            "created_at": doc.get("created_at"),
            "markdown_content": doc.get("markdown_content"),
        }
    return None


async def list_prd_versions(project_id: str) -> list[dict]:
    db = get_db()
    # Only include versions that look like finalized PRD markdown.
    cursor = db.prd_versions.find({"project_id": project_id}).sort("version_number", -1)
    versions: list[dict] = []
    async for doc in cursor:
        if not _looks_like_prd_markdown(str(doc.get("markdown_content") or "")):
            continue
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
    if not _looks_like_prd_markdown(str(doc.get("markdown_content") or "")):
        return None
    return {
        "project_id": doc.get("project_id"),
        "version_number": doc.get("version_number"),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "markdown_content": doc.get("markdown_content"),
    }
