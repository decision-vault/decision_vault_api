from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import settings
from app.db.mongo import get_db
from app.services.llm_usage_service import log_llm_usage

PARTICIPANT_NODE_TYPE = "sequenceParticipant"
VECTOR_DIM = 256
MAX_HISTORY_CANDIDATES = 180
MAX_RETRIEVED_CHUNKS = 6
MAX_CHUNK_CHARS = 720
CHUNK_OVERLAP = 120
PROJECT_VECTOR_COLLECTION = "project_vector_chunks"


class UsecaseNodeData(BaseModel):
    name: str


class UsecaseNode(BaseModel):
    id: str
    type: str = PARTICIPANT_NODE_TYPE
    position: dict[str, float] = Field(default_factory=lambda: {"x": 100.0, "y": 100.0})
    data: UsecaseNodeData

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        node = dict(value)
        node["type"] = PARTICIPANT_NODE_TYPE
        pos = node.get("position")
        if isinstance(pos, dict):
            node["position"] = {"x": float(pos.get("x", 100)), "y": float(pos.get("y", 100))}
        if not isinstance(node.get("data"), dict):
            node["data"] = {"name": str(node.get("name") or node.get("id") or "Participant")}
        elif not node["data"].get("name"):
            node["data"]["name"] = str(node.get("id") or "Participant")
        return node


class UsecaseEdgeData(BaseModel):
    label: str = ""


class UsecaseEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    data: UsecaseEdgeData = Field(default_factory=UsecaseEdgeData)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        edge = dict(value)
        label = str(edge.get("label") or "")
        data = edge.get("data")
        if not isinstance(data, dict):
            edge["data"] = {"label": label}
        elif not data.get("label") and label:
            data["label"] = label
            edge["data"] = data
        if not edge.get("label") and edge.get("data", {}).get("label"):
            edge["label"] = str(edge["data"]["label"])
        return edge


class UsecaseFlowOutput(BaseModel):
    nodes: list[UsecaseNode] = Field(default_factory=list)
    edges: list[UsecaseEdge] = Field(default_factory=list)
    summary: str = "Use case diagram updated."


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


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for block in parts:
            block = block.strip()
            if not block:
                continue
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{") and "}" in block:
                stripped = block
                break
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parse_json(raw: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(raw)
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = ast.literal_eval(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Usecase flow output is not a JSON object")
    return parsed


def _post_process_output(data: dict[str, Any]) -> dict[str, Any]:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    normalized_nodes = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"participant_{idx}")
        normalized_nodes.append(
            {
                "id": node_id,
                "type": PARTICIPANT_NODE_TYPE,
                "position": node.get("position") or {"x": 80 + (idx * 220), "y": 100},
                "data": {"name": str((node.get("data") or {}).get("name") or node_id)},
            }
        )
    node_ids = {n["id"] for n in normalized_nodes}
    normalized_edges = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target or source not in node_ids or target not in node_ids:
            continue
        label = str(edge.get("label") or (edge.get("data") or {}).get("label") or "")
        normalized_edges.append(
            {
                "id": str(edge.get("id") or f"e-{source}-{target}-{idx}"),
                "source": source,
                "target": target,
                "label": label,
                "data": {"label": label},
            }
        )

    if not data.get("summary"):
        data["summary"] = f"Use case diagram updated with {len(normalized_nodes)} participants and {len(normalized_edges)} interactions."
    data["nodes"] = normalized_nodes
    data["edges"] = normalized_edges
    return data


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


def _build_retrieval_query(
    *,
    user_request: str,
    project_name: Any,
    problem_statement: Any,
    desired_features: Any,
) -> str:
    features = desired_features if isinstance(desired_features, list) else []
    return "\n".join(
        [
            f"user_request: {str(user_request or '').strip()}",
            f"project_name: {str(project_name or '').strip()}",
            f"problem_statement: {str(problem_statement or '').strip()}",
            "features: " + ", ".join(str(x).strip() for x in features if str(x).strip()),
        ]
    ).strip()


def _usecase_result_to_text(output: dict[str, Any]) -> str:
    nodes = output.get("nodes") or []
    edges = output.get("edges") or []
    participants = [str((n.get("data") or {}).get("name") or n.get("id") or "").strip() for n in nodes if isinstance(n, dict)]
    interactions = [str(e.get("label") or "").strip() for e in edges if isinstance(e, dict) and str(e.get("label") or "").strip()]
    sections = [
        f"Summary: {str(output.get('summary') or '').strip()}",
        "Participants: " + ", ".join(p for p in participants if p),
        "Interactions: " + " | ".join(interactions),
    ]
    return "\n".join(s for s in sections if s.strip())


async def _retrieve_usecase_chunks(
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


def _to_oid(value: str | None) -> ObjectId | None:
    if not value:
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


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
        usecase_text = _usecase_result_to_text(
            {
                "nodes": usecase_doc.get("nodes") or [],
                "edges": usecase_doc.get("edges") or [],
                "summary": usecase_doc.get("summary") or "",
            }
        )
        if usecase_text:
            await _upsert_project_doc_chunks(
                tenant_id=tenant_id,
                project_id=project_id,
                source_type="usecase",
                source_id=str(usecase_doc.get("_id")),
                source_version=int(usecase_doc.get("version") or 1),
                text=usecase_text,
            )


async def _store_usecase_chunks(
    *,
    tenant_id: str,
    project_id: str,
    intake_id: str,
    run_id: str,
    output: dict[str, Any],
) -> None:
    text = _usecase_result_to_text(output)
    if not text.strip():
        return
    version_seed = int(_utcnow().timestamp())
    source_id = f"{intake_id}:{run_id or 'direct'}"
    await _upsert_project_doc_chunks(
        tenant_id=tenant_id,
        project_id=project_id,
        source_type="usecase",
        source_id=source_id,
        source_version=version_seed,
        text=text,
    )


async def _invoke_llm(prompt: str, output_tokens: int) -> tuple[str, int, str]:
    model_name, api_key, base_url, provider = _provider_config()
    if not api_key:
        raise ValueError("LLM API key not configured")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        top_p=0.8,
        max_tokens=output_tokens,
        api_key=api_key,
        base_url=_normalize_openai_base_url(base_url, provider),
    )
    msg = await llm.ainvoke(prompt)
    text = (getattr(msg, "content", "") or "").strip()
    meta = getattr(msg, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    total_tokens = int(usage.get("total_tokens") or 0) or max(1, len((prompt + text)) // 4)
    return text, total_tokens, model_name


async def generate_usecase_flow(
    *,
    tenant_id: str,
    project_id: str,
    intake_id: str,
    structured: dict[str, Any],
    current_nodes: list[dict[str, Any]],
    current_edges: list[dict[str, Any]],
    user_request: str,
    latest_sdd_content: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    await _sync_project_knowledge_chunks(tenant_id=tenant_id, project_id=project_id)

    sdd_excerpt = (latest_sdd_content or "").strip()
    if len(sdd_excerpt) > 12000:
        sdd_excerpt = sdd_excerpt[:12000]
    retrieval_query = _build_retrieval_query(
        user_request=user_request,
        project_name=structured.get("project_name"),
        problem_statement=structured.get("problem_statement"),
        desired_features=structured.get("desired_features") or [],
    )
    retrieved_chunks = await _retrieve_usecase_chunks(
        tenant_id=tenant_id,
        project_id=project_id,
        query_text=retrieval_query,
    )

    payload = {
        "project_name": structured.get("project_name"),
        "problem_statement": structured.get("problem_statement"),
        "desired_features": structured.get("desired_features") or [],
        "latest_system_design_context": sdd_excerpt,
        "retrieved_usecase_knowledge_chunks": retrieved_chunks,
        "current_nodes": current_nodes,
        "current_edges": current_edges,
        "user_request": user_request,
    }
    input_json = json.dumps(payload, ensure_ascii=False)
    base_prompt = (
        "You are a systems analyst assistant. "
        "Build/update a use-case interaction diagram for React Flow. "
        "Return JSON only with keys: nodes, edges, summary. "
        "No markdown, no extra keys. "
        "Node shape: {id,type:'sequenceParticipant',position:{x,y},data:{name}}. "
        "Edge shape: {id,source,target,label,data:{label}}. "
        "Represent actors/services/components as participants and API/process interactions as edges with labels. "
        "Keep existing IDs stable when possible.\n\n"
        f"Input:\n{input_json}"
    )

    last_error: Exception | None = None
    prompt = base_prompt
    last_raw = ""
    for attempt in range(3):
        raw, total_tokens, model_name = await _invoke_llm(prompt, 1200)
        last_raw = raw
        try:
            parsed = _parse_json(raw)
            result = UsecaseFlowOutput.model_validate(parsed)
            output = _post_process_output(result.model_dump())
            await log_llm_usage(
                feature="usecase_flow:generate",
                model=model_name,
                input_tokens=max(1, len(input_json) // 4),
                output_tokens=max(1, total_tokens - max(1, len(input_json) // 4)),
                tenant_id=tenant_id,
                stage_name="usecase_flow",
                retry_count=attempt,
            )
            await _store_usecase_chunks(
                tenant_id=tenant_id,
                project_id=project_id,
                intake_id=intake_id,
                run_id=run_id or "",
                output=output,
            )
            return output
        except (ValidationError, ValueError, SyntaxError) as exc:
            last_error = exc
            prompt = (
                f"{base_prompt}\n\n"
                "Previous output invalid. Return ONLY valid JSON with keys nodes, edges, summary.\n"
                f"Validation error: {str(exc)}\n"
                f"Previous output:\n{last_raw}"
            )
    raise ValueError(f"usecase_flow failed: {last_error}")
