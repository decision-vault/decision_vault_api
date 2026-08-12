"""Project knowledge base service.

Document-first KB with a derived decision layer:

- ``knowledge_chunks``  — cleaned, chunked document bodies (ingestion unit).
- ``decision_records``  — decisions extracted from those documents (derived
  layer), always carrying a link back to their source document.

Retrieval is lexical (BM25-lite) so grounded answers work with zero additional
infrastructure — no embedding model required. The scoring is computed in
Python over a project's chunk set, which is fine at this scale; it can be
swapped for a vector store without touching the callers.
"""
from __future__ import annotations

import html
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db

logger = logging.getLogger("decisionvault.knowledge")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 400

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9_\-]+")
HTML_ENTITIES_RE = re.compile(r"&[#\w]+;")

# Instruction block injected into the chat as a system message so the agent
# answers only from the knowledge base and stays honest when it has nothing.
GROUNDING_INSTRUCTION = """You are answering as Kavi using the project knowledge base below.

The passages above were retrieved from the project's documents. Use them as your PRIMARY factual basis.

Rules:
- Base every factual claim on the retrieved passages, and cite the source inline as [Source 1], [Source 2], etc.
- If the retrieved passages do NOT contain the information the user asked about, say clearly: "We haven't documented this yet."
  Then briefly offer to create a new decision record to capture it.
- Never invent details, decisions, or context that are not present in the passages.
- When a known decision is listed, you may reference it and cite its source."""


def _utcnow():
    return datetime.utcnow()


def _clean_text(text: str) -> str:
    """Strip HTML markup from document bodies and normalize whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_ENTITIES_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split cleaned text into overlapping chunks on word boundaries."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks[:MAX_CHUNKS_PER_DOC]


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def _token_counts(tokens: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _score_chunks(query_tokens: List[str], chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
    """BM25-lite scoring over a project's chunk set (computed in Python)."""
    if not query_tokens or not chunks:
        return []
    tokenized = [_tokenize(c["text"]) for c in chunks]
    counts = [_token_counts(t) for t in tokenized]
    n = len(chunks)
    avgdl = max(1.0, sum(len(t) for t in tokenized) / n)
    df: Dict[str, int] = {}
    for c in counts:
        for t in c:
            df[t] = df.get(t, 0) + 1
    k1, b = 1.2, 0.75
    q_set = set(query_tokens)
    scored = []
    for i, c in enumerate(chunks):
        dl = len(tokenized[i])
        score = 0.0
        for t in q_set:
            tf = counts[i].get(t, 0)
            if not tf:
                continue
            idf = math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _oid_matches(value) -> List[Any]:
    """Return candidate query values for a project/tenant id that may be stored
    as ObjectId, string, or both."""
    s = str(value)
    candidates = [s]
    if ObjectId.is_valid(s):
        try:
            candidates.append(ObjectId(s))
        except Exception:
            pass
    return candidates


async def _project_workspace_ids(tenant_id: str, project_id: str) -> List[str]:
    db = get_db()
    cursor = db.workspaces.find({"tenant_id": {"$in": _oid_matches(tenant_id)}, "project_id": {"$in": _oid_matches(project_id)}})
    return [str(ws["_id"]) for ws in await cursor.to_list(length=200)]


async def _project_documents(tenant_id: str, project_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    ws_ids = await _project_workspace_ids(tenant_id, project_id)
    if not ws_ids:
        return []
    cursor = db.documents.find({"tenant_id": tenant_id, "workspace_id": {"$in": ws_ids}})
    docs = await cursor.to_list(length=500)
    # Direct-link docs (document_id == project doc) fallback: documents that
    # carry a project_id field directly.
    direct = await db.documents.find({"tenant_id": tenant_id, "project_id": {"$in": _oid_matches(project_id)}}).to_list(length=200)
    seen = {str(d["_id"]) for d in docs}
    docs.extend(d for d in direct if str(d["_id"]) not in seen)
    return docs


async def ensure_indexed(tenant_id: str, project_id: str) -> int:
    """Incremental freshness: re-chunk only documents whose ``updated_at`` is
    newer than their newest chunk. Keeps the KB current on every chat step
    without full re-indexing. Returns total chunk count."""
    db = get_db()
    docs = await _project_documents(tenant_id, project_id)
    for doc in docs:
        doc_id = str(doc["_id"])
        updated_at = doc.get("updated_at")
        latest = await db.knowledge_chunks.find_one(
            {"tenant_id": tenant_id, "project_id": project_id, "document_id": doc_id},
            sort=[("chunk_index", -1)],
        )
        stale = latest is None or (
            updated_at is not None and latest.get("created_at") is not None and updated_at > latest["created_at"]
        )
        if not stale:
            continue
        title = doc.get("title") or "Untitled document"
        text = _clean_text(doc.get("body", ""))
        chunks = _chunk_text(text) if text else []
        await db.knowledge_chunks.delete_many(
            {"tenant_id": tenant_id, "project_id": project_id, "document_id": doc_id}
        )
        if chunks:
            await db.knowledge_chunks.insert_many([
                {
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "document_id": doc_id,
                    "source_title": title,
                    "chunk_index": i,
                    "text": chunk,
                    "created_at": _utcnow(),
                }
                for i, chunk in enumerate(chunks)
            ])
    return await db.knowledge_chunks.count_documents(
        {"tenant_id": tenant_id, "project_id": project_id}
    )


async def index_project(tenant_id: str, project_id: str) -> Dict[str, Any]:
    """Chunk every document in a project's workspaces into ``knowledge_chunks``.
    Idempotent: deletes prior chunks for the project and reinserts."""
    db = get_db()
    docs = await _project_documents(tenant_id, project_id)
    chunks_created = 0
    sources = []
    for doc in docs:
        doc_id = str(doc["_id"])
        title = doc.get("title") or "Untitled document"
        text = _clean_text(doc.get("body", ""))
        if not text:
            continue
        chunks = _chunk_text(text)
        if not chunks:
            continue
        await db.knowledge_chunks.delete_many({"tenant_id": tenant_id, "project_id": project_id, "document_id": doc_id})
        if chunks:
            await db.knowledge_chunks.insert_many([
                {
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "document_id": doc_id,
                    "source_title": title,
                    "chunk_index": i,
                    "text": chunk,
                    "created_at": _utcnow(),
                }
                for i, chunk in enumerate(chunks)
            ])
        chunks_created += len(chunks)
        sources.append({"document_id": doc_id, "title": title, "chunks": len(chunks)})

    return {
        "indexed_documents": len(sources),
        "chunks_created": chunks_created,
        "sources": sources,
    }


async def search(tenant_id: str, project_id: str, query: str, top_k: int = 4) -> Dict[str, Any]:
    """Retrieve the most relevant chunks and decision records for a query."""
    db = get_db()
    query_tokens = _tokenize(query)

    results = []
    if query_tokens:
        chunks = await db.knowledge_chunks.find(
            {"tenant_id": tenant_id, "project_id": project_id}
        ).to_list(length=MAX_CHUNKS_PER_DOC * 10)
        scored = _score_chunks(query_tokens, chunks)[:top_k]
        results = [
            {
                "document_id": c["document_id"],
                "source_title": c["source_title"],
                "chunk_text": c["text"],
                "score": round(score, 4),
            }
            for c, score in scored
        ]

    decisions = []
    decision_cursor = db.decision_records.find(
        {"tenant_id": tenant_id, "project_id": project_id}
    )
    for rec in await decision_cursor.to_list(length=200):
        haystack = _tokenize(f"{rec.get('title','')} {rec.get('context','')} {rec.get('choice','')}")
        overlap = len(set(query_tokens) & set(haystack))
        if overlap > 0:
            decisions.append({
                "title": rec.get("title", ""),
                "context": rec.get("context", ""),
                "alternatives": rec.get("alternatives", []),
                "choice": rec.get("choice", ""),
                "rationale": rec.get("rationale", ""),
                "outcome": rec.get("outcome", ""),
                "source_document_id": rec.get("document_id", ""),
                "source_title": rec.get("source_title", ""),
            })

    return {"query": query, "results": results, "decisions": decisions}


async def store_decision_records(
    tenant_id: str,
    project_id: str,
    document_id: str,
    source_title: str,
    records: List[Dict[str, Any]],
) -> int:
    """Persist extracted decision records, de-duplicating by (doc, title)."""
    db = get_db()
    stored = 0
    for rec in records:
        title = (rec.get("title") or "").strip()
        if not title:
            continue
        existing = await db.decision_records.find_one({
            "tenant_id": tenant_id,
            "project_id": project_id,
            "document_id": document_id,
            "title": title,
        })
        record = {
            "tenant_id": tenant_id,
            "project_id": project_id,
            "document_id": document_id,
            "source_title": source_title,
            "title": title,
            "context": rec.get("context", ""),
            "alternatives": rec.get("alternatives", []),
            "choice": rec.get("choice", ""),
            "rationale": rec.get("rationale", ""),
            "outcome": rec.get("outcome", ""),
            "updated_at": _utcnow(),
        }
        if existing:
            await db.decision_records.update_one({"_id": existing["_id"]}, {"$set": record})
        else:
            record["created_at"] = _utcnow()
            await db.decision_records.insert_one(record)
        stored += 1
    return stored


async def extract_decisions_for_project(tenant_id: str, project_id: str) -> int:
    """Derive the decision layer: run the extraction agent over each indexed
    document and persist the resulting decision records."""
    db = get_db()
    docs = await _project_documents(tenant_id, project_id)
    agent_url = f"{settings.langgraph_url}/workflow/knowledge/extract-decisions"
    total = 0
    async with httpx.AsyncClient(timeout=300.0) as client:
        for doc in docs:
            doc_id = str(doc["_id"])
            title = doc.get("title") or "Untitled document"
            text = _clean_text(doc.get("body", ""))[:8000]
            if not text:
                continue
            try:
                resp = await client.post(agent_url, json={"text": text, "product_name": title})
                resp.raise_for_status()
                records = resp.json().get("decisions", [])
                if records:
                    total += await store_decision_records(
                        tenant_id, project_id, doc_id, title, records
                    )
            except Exception as exc:
                logger.warning(f"Decision extraction failed for {doc_id}: {exc}")
    return total


async def build_grounded_context(
    tenant_id: str,
    project_id: str,
    query: str,
    top_k: int = 4,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Build the retrieval context injected into the chat pipeline.

    Returns ``(system_messages, citations)``:
    - system_messages: instruction + retrieved passages (+ known decisions)
    - citations: source list the UI renders under the answer
    """
    result = await search(tenant_id, project_id, query, top_k=top_k)
    chunks = result["results"]
    decisions = result["decisions"]

    citations: List[Dict[str, Any]] = []
    passages: List[str] = []
    for i, c in enumerate(chunks, start=1):
        passages.append(
            f"[Source {i}] (from {c['source_title']}): {c['chunk_text']}"
        )
        citations.append({
            "source_title": c["source_title"],
            "document_id": c["document_id"],
            "chunk_text": c["chunk_text"][:300],
            "score": c["score"],
        })

    decision_block = ""
    if decisions:
        lines = []
        for d in decisions:
            lines.append(f"- {d['title']} → {d['choice']} (source: {d['source_title']})")
            citations.append({
                "source_title": d["source_title"],
                "document_id": d["source_document_id"],
                "chunk_text": f"Decision: {d['title']}",
                "score": 1.0,
            })
        decision_block = "\n\nKnown decisions in this project:\n" + "\n".join(lines)

    if not passages:
        passages.append("[Source 1] (no matching passages retrieved)")

    context_msg = (
        "Retrieved from the project knowledge base:\n\n"
        + "\n\n".join(passages)
        + decision_block
        + "\n\n"
        + GROUNDING_INSTRUCTION
    )
    return ([{"role": "system", "content": context_msg}], citations)
