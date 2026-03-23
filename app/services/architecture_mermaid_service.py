from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.db.mongo import get_db
from app.services.llm_usage_service import log_llm_usage

VECTOR_DIM = 256
MAX_HISTORY_CANDIDATES = 180
MAX_RETRIEVED_CHUNKS = 6
MAX_CHUNK_CHARS = 720
CHUNK_OVERLAP = 120
PROJECT_VECTOR_COLLECTION = "project_vector_chunks"


def _provider_config() -> tuple[str, str, str | None, str]:
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "lmstudio":
        return (
            settings.lmstudio_model or settings.llm_model,
            settings.llm_api_key or "lm-studio",
            settings.lmstudio_base_url,
            "lmstudio",
        )
    if provider == "huggingface":
        return (
            settings.hf_openai_model or settings.llm_model,
            settings.hf_api_token,
            settings.hf_router_base_url,
            "huggingface",
        )
    return (settings.llm_model, settings.llm_api_key, settings.llm_base_url, "default")


def _normalize_openai_base_url(base_url: str | None, provider: str) -> str | None:
    if not base_url:
        return base_url
    normalized = base_url.rstrip("/")
    if provider == "lmstudio":
        if normalized.endswith("/api/v1"):
            normalized = normalized[: -len("/api/v1")] + "/v1"
        elif not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
    return normalized


def _extract_mermaid(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    fenced = re.search(r"```(?:mermaid)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    lines = raw.splitlines()
    start = -1
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("flowchart") or s.startswith("graph "):
            start = i
            break
    if start >= 0:
        raw = "\n".join(lines[start:]).strip()

    if not raw:
        return ""
    first = raw.splitlines()[0].strip().lower()
    if not (first.startswith("flowchart") or first.startswith("graph ")):
        return ""
    return raw


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
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


def _to_oid(value: str | None) -> ObjectId | None:
    if not value:
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


async def _upsert_project_doc_chunks(
    *,
    tenant_id: str,
    project_id: str,
    source_type: str,
    source_id: str,
    source_version: int,
    text: str,
) -> None:
    chunks = _chunk_text(text)
    if not chunks:
        return
    now = _utcnow()
    db = get_db()
    coll = db[PROJECT_VECTOR_COLLECTION]
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


def _architecture_result_to_text(*, mermaid: str, summary: str, user_request: str) -> str:
    return "\n".join(
        [
            f"request: {str(user_request or '').strip()}",
            f"summary: {str(summary or '').strip()}",
            "mermaid:",
            str(mermaid or "").strip(),
        ]
    ).strip()


async def _sync_project_knowledge_chunks(*, tenant_id: str, project_id: str) -> None:
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
        await _upsert_project_doc_chunks(
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
        await _upsert_project_doc_chunks(
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
            await _upsert_project_doc_chunks(
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
            await _upsert_project_doc_chunks(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="usecase",
                source_id=str(usecase_doc.get("_id")),
                source_version=int(usecase_doc.get("version") or 1),
                text=usecase_text,
            )

    architecture_doc = await db.architecture_diagram_documents.find_one(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        sort=[("generated_at", -1)],
    )
    if architecture_doc:
        architecture_text = _architecture_result_to_text(
            mermaid=str(architecture_doc.get("mermaid") or ""),
            summary=str(architecture_doc.get("summary") or ""),
            user_request=str(architecture_doc.get("view") or "architecture"),
        )
        if architecture_text:
            await _upsert_project_doc_chunks(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="architecture",
                source_id=str(architecture_doc.get("_id")),
                source_version=int(architecture_doc.get("version") or 1),
                text=architecture_text,
            )


async def _retrieve_architecture_chunks(
    *,
    tenant_id: str,
    project_id: str,
    query_text: str,
    top_k: int = MAX_RETRIEVED_CHUNKS,
) -> list[str]:
    db = get_db()
    docs = (
        await db[PROJECT_VECTOR_COLLECTION].find(
            {"tenant_id": tenant_id, "project_id": project_id},
            {"_id": 0, "chunk_text": 1, "embedding": 1},
        )
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


async def _store_architecture_chunks(
    *,
    tenant_id: str,
    project_id: str,
    intake_id: str,
    run_id: str,
    user_request: str,
    mermaid: str,
    summary: str,
) -> None:
    text = _architecture_result_to_text(mermaid=mermaid, summary=summary, user_request=user_request)
    if not text.strip():
        return
    version_seed = int(_utcnow().timestamp())
    source_id = f"{intake_id}:{run_id or 'direct'}"
    await _upsert_project_doc_chunks(
        tenant_id=tenant_id,
        project_id=project_id,
        source_type="architecture",
        source_id=source_id,
        source_version=version_seed,
        text=text,
    )


def _truncate(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


async def generate_architecture_mermaid(
    *,
    user_request: str,
    structured: dict | None = None,
    latest_sdd_content: str = "",
    tenant_id: str | None = None,
    project_id: str | None = None,
    intake_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    if tenant_id and project_id:
        await _sync_project_knowledge_chunks(tenant_id=tenant_id, project_id=project_id)

    retrieved_chunks: list[str] = []
    if tenant_id and project_id:
        query_parts = [
            str(user_request or "").strip(),
            str((structured or {}).get("project_name") or "").strip(),
            str((structured or {}).get("problem_statement") or "").strip(),
            ", ".join(str(x).strip() for x in ((structured or {}).get("desired_features") or []) if str(x).strip()),
        ]
        retrieved_chunks = await _retrieve_architecture_chunks(
            tenant_id=tenant_id,
            project_id=project_id,
            query_text="\n".join(q for q in query_parts if q),
        )

    model, api_key, base_url, provider = _provider_config()
    normalized_base_url = _normalize_openai_base_url(base_url, provider)
    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        api_key=api_key,
        base_url=normalized_base_url,
    )

    structured_json = json.dumps(structured or {}, ensure_ascii=True, indent=2)
    prompt = f"""
You are an expert software architect. Generate one Mermaid architecture diagram.

Return ONLY Mermaid flowchart code. No markdown fences. No explanation.
The first line must be either:
- flowchart TB
- flowchart LR
- graph TB
- graph LR

Constraints:
- Keep node ids simple alphanumeric/underscore.
- Use clear labels in brackets.
- Include the most relevant components only (8-18 nodes).
- Include directional edges with meaningful labels when useful.
- Prefer correctness based on the provided requirements and system design.

User request:
{_truncate(user_request, 1200)}

Structured requirements JSON:
{_truncate(structured_json, 9000)}

Latest system design context:
{_truncate(latest_sdd_content, 7000)}

Retrieved project knowledge chunks:
{_truncate(json.dumps(retrieved_chunks, ensure_ascii=True), 9000)}
""".strip()

    response = await asyncio.to_thread(llm.invoke, prompt)
    mermaid = _extract_mermaid(getattr(response, "content", "") or "")
    if not mermaid:
        raise ValueError("Model did not return valid Mermaid flowchart output.")

    summary = f"Architecture diagram generated from requirements and request: {user_request.strip() or 'default'}"
    input_tokens = max(1, len(prompt) // 4)
    output_tokens = max(1, len(mermaid) // 4)
    await log_llm_usage(
        feature="architecture_diagram:generate",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tenant_id=tenant_id,
        stage_name="architecture_diagram",
        retry_count=0,
    )
    if tenant_id and project_id and intake_id:
        await _store_architecture_chunks(
            tenant_id=tenant_id,
            project_id=project_id,
            intake_id=intake_id,
            run_id=run_id or "",
            user_request=user_request,
            mermaid=mermaid,
            summary=summary,
        )
    return {
        "mermaid": mermaid,
        "summary": summary,
        "view": "llm_generated",
    }
