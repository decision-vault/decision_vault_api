from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.db.mongo import get_db

VECTOR_DIM = 256
MAX_HISTORY_CANDIDATES = 180
PROJECT_VECTOR_COLLECTION = "project_vector_chunks"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_oid(value: str | None) -> ObjectId | None:
    if not value:
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{2,}", text.lower())


def _embed_text_local(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        h = hash(token)
        idx = abs(h) % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def chunk_text(text: str, max_chars: int = 720, overlap: int = 120) -> list[str]:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def _schema_doc_to_text(doc: dict[str, Any]) -> str:
    nodes = doc.get("nodes") or []
    edges = doc.get("edges") or []
    parts: list[str] = [f"summary: {str(doc.get('summary') or '').strip()}"]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        table = str((node.get("data") or {}).get("tableName") or "").strip()
        cols = (node.get("data") or {}).get("columns") or []
        if not table:
            continue
        col_names = [str(c.get("name") or "").strip() for c in cols if isinstance(c, dict)]
        parts.append(f"table {table}: {', '.join(c for c in col_names if c)}")
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source") or "").strip()
        tgt = str(edge.get("target") or "").strip()
        if src and tgt:
            parts.append(f"relation {src} -> {tgt}")
    return "\n".join(parts).strip()


def _usecase_doc_to_text(doc: dict[str, Any]) -> str:
    nodes = doc.get("nodes") or []
    edges = doc.get("edges") or []
    participants = [str((n.get("data") or {}).get("name") or n.get("id") or "").strip() for n in nodes if isinstance(n, dict)]
    interactions = [str(e.get("label") or "").strip() for e in edges if isinstance(e, dict) and str(e.get("label") or "").strip()]
    sections = [
        f"Summary: {str(doc.get('summary') or '').strip()}",
        "Participants: " + ", ".join(p for p in participants if p),
        "Interactions: " + " | ".join(interactions),
    ]
    return "\n".join(s for s in sections if s.strip())


async def store_project_source_text(
    *,
    tenant_id: str,
    project_id: str,
    source_type: str,
    source_id: str,
    source_version: int,
    text: str,
) -> None:
    chunks = chunk_text(text)
    if not chunks:
        return
    db = get_db()
    coll = db[PROJECT_VECTOR_COLLECTION]
    now = _utcnow()
    for idx, chunk in enumerate(chunks):
        stable_key = f"{tenant_id}:{project_id}:{source_type}:{source_id}:{source_version}:{idx}"
        _id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
        await coll.update_one(
            {"_id": _id},
            {
                "$set": {
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_version": int(source_version),
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "embedding": _embed_text_local(chunk),
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )


async def retrieve_project_knowledge_chunks(
    *,
    tenant_id: str,
    project_id: str,
    query_text: str,
    top_k: int = 6,
) -> list[str]:
    db = get_db()
    docs = (
        await db[PROJECT_VECTOR_COLLECTION]
        .find({"tenant_id": tenant_id, "project_id": project_id}, {"_id": 0, "chunk_text": 1, "embedding": 1})
        .sort("created_at", -1)
        .limit(MAX_HISTORY_CANDIDATES)
        .to_list(length=MAX_HISTORY_CANDIDATES)
    )
    if not docs:
        return []
    query_vec = _embed_text_local(query_text)
    scored: list[tuple[float, str]] = []
    for doc in docs:
        text = str(doc.get("chunk_text") or "").strip()
        emb = doc.get("embedding") or []
        if not text or not isinstance(emb, list):
            continue
        vec = []
        for x in emb[:VECTOR_DIM]:
            try:
                vec.append(float(x))
            except Exception:
                vec.append(0.0)
        if len(vec) != VECTOR_DIM:
            continue
        score = _cosine_similarity(query_vec, vec)
        if score > 0:
            scored.append((score, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    results: list[str] = []
    for _, txt in scored:
        if txt in seen:
            continue
        seen.add(txt)
        results.append(txt)
        if len(results) >= top_k:
            break
    return results


async def sync_project_knowledge_chunks(*, tenant_id: str, project_id: str) -> None:
    db = get_db()
    tenant_oid = _to_oid(tenant_id)
    project_oid = _to_oid(project_id)
    if not tenant_oid or not project_oid:
        return

    prd_doc = await db.prd_documents.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        sort=[("generated_at", -1)],
    )
    if prd_doc and str(prd_doc.get("content") or "").strip():
        await store_project_source_text(
            tenant_id=tenant_id,
            project_id=project_id,
            source_type="prd",
            source_id=str(prd_doc.get("_id")),
            source_version=int(prd_doc.get("version") or 1),
            text=str(prd_doc.get("content") or ""),
        )

    sdd_doc = await db.system_design_documents.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        sort=[("generated_at", -1)],
    )
    if sdd_doc and str(sdd_doc.get("content") or "").strip():
        await store_project_source_text(
            tenant_id=tenant_id,
            project_id=project_id,
            source_type="sdd",
            source_id=str(sdd_doc.get("_id")),
            source_version=int(sdd_doc.get("version") or 1),
            text=str(sdd_doc.get("content") or ""),
        )

    schema_doc = await db.schema_flow_documents.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        sort=[("generated_at", -1)],
    )
    if schema_doc:
        schema_text = _schema_doc_to_text(schema_doc)
        if schema_text:
            await store_project_source_text(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="schema",
                source_id=str(schema_doc.get("_id")),
                source_version=int(schema_doc.get("version") or 1),
                text=schema_text,
            )

    usecase_doc = await db.usecase_flow_documents.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        sort=[("generated_at", -1)],
    )
    if usecase_doc:
        usecase_text = _usecase_doc_to_text(usecase_doc)
        if usecase_text:
            await store_project_source_text(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="usecase",
                source_id=str(usecase_doc.get("_id")),
                source_version=int(usecase_doc.get("version") or 1),
                text=usecase_text,
            )
