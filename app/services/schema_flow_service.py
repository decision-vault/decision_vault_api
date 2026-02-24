from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import settings
from app.services.llm_usage_service import log_llm_usage

TABLE_NODE_TYPE = "schemaTable"


class SchemaColumn(BaseModel):
    name: str
    type: str
    primaryKey: bool = False
    unique: bool = False


class SchemaNodeData(BaseModel):
    tableName: str
    columns: list[SchemaColumn] = Field(default_factory=list)


class SchemaNode(BaseModel):
    id: str
    type: str = TABLE_NODE_TYPE
    position: dict[str, float] = Field(default_factory=lambda: {"x": 100.0, "y": 100.0})
    data: SchemaNodeData

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if not normalized.get("type"):
            normalized["type"] = TABLE_NODE_TYPE
        if normalized.get("type") == "table":
            normalized["type"] = TABLE_NODE_TYPE
        pos = normalized.get("position")
        if isinstance(pos, dict):
            normalized["position"] = {"x": float(pos.get("x", 100)), "y": float(pos.get("y", 100))}
        # Accept legacy node shapes where tableName/columns are top-level and "data" is missing.
        if not isinstance(normalized.get("data"), dict):
            table_name = normalized.get("tableName") or normalized.get("table_name")
            columns = normalized.get("columns")
            if not isinstance(columns, list):
                columns = []
            if not table_name:
                table_name = normalized.get("id") or "table"
            normalized["data"] = {
                "tableName": str(table_name),
                "columns": columns,
            }
        else:
            data = dict(normalized["data"])
            if not data.get("tableName"):
                data["tableName"] = (
                    data.get("table_name")
                    or normalized.get("tableName")
                    or normalized.get("id")
                    or "table"
                )
            if not isinstance(data.get("columns"), list):
                data["columns"] = []
            normalized["data"] = data
        return normalized


class SchemaEdge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class SchemaFlowOutput(BaseModel):
    nodes: list[SchemaNode] = Field(default_factory=list)
    edges: list[SchemaEdge] = Field(default_factory=list)
    summary: str = "Schema updated."


def _to_singular(name: str) -> str:
    n = (name or "").strip().lower()
    if n.endswith("ies") and len(n) > 3:
        return n[:-3] + "y"
    if n.endswith("s") and len(n) > 1:
        return n[:-1]
    return n


def _normalize_column_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    mapping = {
        "string": "text",
        "str": "text",
        "varchar": "text",
        "character varying": "text",
        "integer": "int",
        "int4": "int",
        "bigint": "bigint",
        "number": "numeric",
        "float": "numeric",
        "double": "numeric",
        "bool": "boolean",
        "datetime": "timestamp",
        "date_time": "timestamp",
        "timestampz": "timestamp",
        "json": "jsonb",
        "object": "jsonb",
    }
    return mapping.get(value, value or "text")


def _sanitize_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or TABLE_NODE_TYPE).strip()
        # Keep database table nodes only.
        if node_type not in {TABLE_NODE_TYPE, "table"}:
            continue
        data = node.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        table_name = str(data.get("tableName") or node.get("id") or f"table_{idx}").strip()
        raw_columns = data.get("columns") if isinstance(data.get("columns"), list) else []
        seen: set[str] = set()
        columns: list[dict[str, Any]] = []
        for col in raw_columns:
            if not isinstance(col, dict):
                continue
            col_name = str(col.get("name") or "").strip()
            if not col_name:
                continue
            key = col_name.lower()
            if key in seen:
                continue
            seen.add(key)
            col_type = _normalize_column_type(str(col.get("type") or "text"))
            if key.endswith("_id") and col_type in {"int", "bigint", "text"}:
                # prefer UUID FKs for this app schema unless explicitly constrained otherwise
                col_type = "uuid"
            columns.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "primaryKey": bool(col.get("primaryKey")),
                    "unique": bool(col.get("unique")),
                }
            )
        cleaned.append(
            {
                "id": str(node.get("id") or table_name.lower().replace(" ", "_")),
                "type": TABLE_NODE_TYPE,
                "position": node.get("position") or {"x": 100.0 + idx * 80.0, "y": 100.0},
                "data": {"tableName": table_name, "columns": columns},
            }
        )
    return cleaned


def _ensure_pk_columns(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        data = node.get("data") or {}
        cols = data.get("columns") or []
        if not isinstance(cols, list) or not cols:
            continue
        has_pk = any(bool(c.get("primaryKey")) for c in cols if isinstance(c, dict))
        if has_pk:
            continue
        for col in cols:
            if isinstance(col, dict) and str(col.get("name") or "").strip().lower() == "id":
                col["primaryKey"] = True
                break


def _ensure_baseline_columns(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        cols = (node.get("data") or {}).get("columns") or []
        if not isinstance(cols, list):
            continue
        names = {str(c.get("name") or "").strip().lower() for c in cols if isinstance(c, dict)}
        if "id" not in names:
            cols.insert(0, {"name": "id", "type": "uuid", "primaryKey": True, "unique": True})
            names.add("id")
        if "created_at" not in names:
            cols.append({"name": "created_at", "type": "timestamp", "primaryKey": False, "unique": False})
        if "updated_at" not in names:
            cols.append({"name": "updated_at", "type": "timestamp", "primaryKey": False, "unique": False})
        (node.get("data") or {})["columns"] = cols


def _infer_relationship_edges(
    nodes: list[dict[str, Any]],
    existing_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_ids = {str(n.get("id")) for n in nodes if n.get("id")}
    table_by_key: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        data = node.get("data") or {}
        table_name = str(data.get("tableName") or node_id)
        if not node_id:
            continue
        keys = {
            _to_singular(table_name),
            _to_singular(node_id),
            _to_singular(table_name).replace("_", ""),
            _to_singular(node_id).replace("_", ""),
        }
        for k in keys:
            if k:
                table_by_key[k] = node

    inferred: list[dict[str, Any]] = []
    seen = {
        (
            str(e.get("source") or ""),
            str(e.get("target") or ""),
            str(e.get("targetHandle") or ""),
        )
        for e in existing_edges
    }
    for target_node in nodes:
        target_id = str(target_node.get("id") or "")
        cols = (target_node.get("data") or {}).get("columns") or []
        if not target_id or not isinstance(cols, list):
            continue
        for col in cols:
            if not isinstance(col, dict):
                continue
            col_name = str(col.get("name") or "").strip()
            col_key = col_name.lower()
            if not col_key.endswith("_id") or col_key == "id":
                continue
            ref = _to_singular(col_key[:-3]).replace("_", "")
            source_node = table_by_key.get(ref)
            source_id = str((source_node or {}).get("id") or "")
            if not source_id or source_id not in node_ids or source_id == target_id:
                continue
            signature = (source_id, target_id, f"{col_name}-in")
            if signature in seen:
                continue
            seen.add(signature)
            inferred.append(
                {
                    "id": f"e-{source_id}-{target_id}-{col_name}",
                    "source": source_id,
                    "target": target_id,
                    "sourceHandle": "id-out",
                    "targetHandle": f"{col_name}-in",
                }
            )
    return inferred


def _sanitize_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = {str(n.get("id")) for n in nodes if n.get("id")}
    clean: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target or source not in node_ids or target not in node_ids or source == target:
            continue
        edge_id = str(edge.get("id") or f"e-{source}-{target}-{idx}")
        if edge_id in seen_ids:
            edge_id = f"{edge_id}-{idx}"
        seen_ids.add(edge_id)
        clean.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "sourceHandle": edge.get("sourceHandle"),
                "targetHandle": edge.get("targetHandle"),
            }
        )
    return clean


def _build_summary(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    table_names = [str((n.get("data") or {}).get("tableName") or n.get("id")) for n in nodes]
    preview = ", ".join(table_names[:8])
    more = "" if len(table_names) <= 8 else f", +{len(table_names) - 8} more"
    return (
        f"Schema Plan Ready\n"
        f"- Tables: {len(nodes)}\n"
        f"- Relationships: {len(edges)}\n"
        f"- Core entities: {preview}{more}\n"
        f"- Next: review foreign keys and add indexes for frequent query paths."
    )


def _post_process_output(data: dict[str, Any]) -> dict[str, Any]:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    nodes = _sanitize_nodes(nodes)
    _ensure_baseline_columns(nodes)
    _ensure_pk_columns(nodes)
    edges = _sanitize_edges(nodes, edges)
    inferred = _infer_relationship_edges(nodes, edges)
    if inferred:
        edges = [*edges, *inferred]
    data["nodes"] = nodes
    data["edges"] = edges
    summary = str(data.get("summary") or "").strip()
    if len(summary) < 20:
        data["summary"] = _build_summary(nodes, edges)
    return data


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
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        end = -1
        for idx, ch in enumerate(stripped[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end > start:
            return stripped[start : end + 1]

    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parse_json(raw: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(raw)
    # light cleanup for common local-model JSON issues
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    def _balance_json_like(text: str) -> str:
        stack: list[str] = []
        out: list[str] = []
        in_string = False
        escape = False
        pairs = {"{": "}", "[": "]"}
        for ch in text:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                continue
            if ch in "{[":
                stack.append(pairs[ch])
                out.append(ch)
                continue
            if ch in "}]":
                if stack and ch == stack[-1]:
                    stack.pop()
                    out.append(ch)
                # skip mismatched/unexpected closing bracket
                continue
            out.append(ch)
        while stack:
            out.append(stack.pop())
        return "".join(out)

    candidate = _balance_json_like(candidate)
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = ast.literal_eval(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Schema flow output is not a JSON object")
    return parsed


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


async def generate_schema_flow(
    *,
    tenant_id: str,
    structured: dict[str, Any],
    current_nodes: list[dict[str, Any]],
    current_edges: list[dict[str, Any]],
    user_request: str,
    latest_sdd_content: str | None = None,
) -> dict[str, Any]:
    sdd_excerpt = (latest_sdd_content or "").strip()
    if len(sdd_excerpt) > 12000:
        sdd_excerpt = sdd_excerpt[:12000]
    payload = {
        "project_name": structured.get("project_name"),
        "desired_features": structured.get("desired_features") or [],
        "constraints": (structured.get("constraints") or {}).get("hard_constraints") or [],
        "latest_system_design_context": sdd_excerpt,
        "current_nodes": current_nodes,
        "current_edges": current_edges,
        "user_request": user_request,
    }
    input_json = json.dumps(payload, ensure_ascii=False)
    base_prompt = (
        "You are a database architect assistant. "
        "Update a React Flow DB schema graph from user request. "
        "Return JSON only with keys: nodes, edges, summary. "
        "Do not include markdown, code fences, or extra keys. "
        "Generate only database tables (no API endpoints, no middleware nodes). "
        "Node shape: {id,type,position:{x,y},data:{tableName,columns:[{name,type,primaryKey,unique}]}}. "
        "Edge shape: {id,source,target,sourceHandle,targetHandle}. "
        "Handle convention: sourceHandle='<column>-out' and targetHandle='<column>-in'. "
        "Use sourceHandle only for primary-key columns on source tables. "
        "Mark primary keys explicitly (usually 'id'). "
        "Each table should include useful fields and audit columns where applicable. "
        "Generate relationship edges for foreign-key columns like '<table>_id'. "
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
            result = SchemaFlowOutput.model_validate(parsed)
            output = _post_process_output(result.model_dump())
            await log_llm_usage(
                feature="schema_flow:generate",
                model=model_name,
                input_tokens=max(1, len(input_json) // 4),
                output_tokens=max(1, total_tokens - max(1, len(input_json) // 4)),
                tenant_id=tenant_id,
                stage_name="schema_flow",
                retry_count=attempt,
            )
            return output
        except (ValidationError, ValueError, SyntaxError) as exc:
            last_error = exc
            prompt = (
                f"{base_prompt}\n\n"
                "Your previous output was invalid.\n"
                "Return ONLY VALID JSON (no prose) with exactly keys: nodes, edges, summary.\n"
                "Use double quotes for all strings. Do not include trailing commas.\n"
                "Keep it compact and syntactically valid.\n"
                f"Validation error: {str(exc)}\n"
                f"Previous output:\n{last_raw}"
            )
            continue
    raise ValueError(f"schema_flow failed: {last_error}")
