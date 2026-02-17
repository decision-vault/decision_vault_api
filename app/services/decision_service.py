from __future__ import annotations

from bson import ObjectId

from pymongo.errors import DuplicateKeyError

from app.db.mongo import get_db
from app.services.decision_embedding_service import embed_and_store_decision


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


async def create_decision_if_new(record: dict) -> bool:
    db = get_db()
    try:
        result = await db.decisions.insert_one(record)
        record["_id"] = result.inserted_id
        await embed_and_store_decision(record)
        return True
    except Exception:
        return False


async def create_custom_decision(payload: dict) -> tuple[bool, str | None, str | None]:
    db = get_db()
    doc = {
        "tenant_id": ObjectId(payload["tenant_id"]),
        "project_id": ObjectId(payload["project_id"]),
        "title": payload["decision_title"],
        "statement": payload["decision_statement"],
        "context": payload.get("context"),
        "source_url": str(payload["source_url"]),
        "timestamp": payload["timestamp"],
        "external_id": payload["external_id"],
        "source": "custom",
        "created_at": payload["timestamp"],
    }
    try:
        result = await db.decisions.insert_one(doc)
        doc["_id"] = result.inserted_id
        await embed_and_store_decision(doc)
        return True, str(result.inserted_id), None
    except DuplicateKeyError:
        return False, None, "duplicate"
    except Exception:
        return False, None, "error"
