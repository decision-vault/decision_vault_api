from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_db
from app.middleware.guard import withGuard


router = APIRouter(prefix="/api/docs", tags=["docs"])


def _as_oid(value: str | None, name: str) -> ObjectId:
    if not value:
        raise HTTPException(status_code=400, detail={name: f"{name} is required"})
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail={name: f"Invalid ObjectId: {value}"})


def _iso(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


DocKind = Literal["prd", "sdd", "schema", "usecase", "sequence", "architecture"]


@router.get("/{doc_id}")
async def get_doc_by_id(
    doc_id: str,
    kind: DocKind,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})

    db = get_db()
    tenant_id = str(user.get("tenant_id") or "")

    if kind == "prd":
        # LLM PRDs live in prd_versions and use a string project_id.
        doc = await db.prd_versions.find_one({"_id": _as_oid(doc_id, "doc_id")})
        if not doc or str(doc.get("project_id") or "") != str(project_id):
            raise HTTPException(status_code=404, detail="PRD doc not found")
        return {
            "doc_id": str(doc.get("_id")),
            "kind": "prd",
            "project_id": str(project_id),
            "version": int(doc.get("version_number") or 1),
            "created_at": _iso(doc.get("created_at")),
            "content": str(doc.get("markdown_content") or ""),
        }

    tenant_oid = _as_oid(tenant_id, "tenant_id")
    project_oid = _as_oid(project_id, "project_id")
    oid = _as_oid(doc_id, "doc_id")

    if kind == "sdd":
        doc = await db.system_design_documents.find_one({"_id": oid, "tenant_id": tenant_oid, "project_id": project_oid})
        if not doc:
            raise HTTPException(status_code=404, detail="SDD doc not found")
        return {
            "doc_id": str(doc.get("_id")),
            "kind": "sdd",
            "project_id": str(project_id),
            "version": int(doc.get("version") or 1),
            "created_at": _iso(doc.get("generated_at") or doc.get("created_at")),
            "content": str(doc.get("content") or ""),
        }

    if kind == "schema":
        doc = await db.schema_flow_documents.find_one({"_id": oid, "tenant_id": tenant_oid, "project_id": project_oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Schema doc not found")
        summary = str(doc.get("summary") or "")
        md = summary.strip() or "# Schema Plan\n\n_No summary available._"
        return {
            "doc_id": str(doc.get("_id")),
            "kind": "schema",
            "project_id": str(project_id),
            "version": int(doc.get("version") or 1),
            "created_at": _iso(doc.get("generated_at") or doc.get("created_at")),
            "content": md,
            "raw": {
                "nodes": doc.get("nodes") or [],
                "edges": doc.get("edges") or [],
                "summary": summary,
            },
        }

    if kind == "usecase":
        doc = await db.usecase_flow_documents.find_one({"_id": oid, "tenant_id": tenant_oid, "project_id": project_oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Usecase doc not found")
        summary = str(doc.get("summary") or "")
        md = summary.strip() or "# Usecase Flow\n\n_No summary available._"
        return {
            "doc_id": str(doc.get("_id")),
            "kind": "usecase",
            "project_id": str(project_id),
            "version": int(doc.get("version") or 1),
            "created_at": _iso(doc.get("generated_at") or doc.get("created_at")),
            "content": md,
            "raw": {
                "nodes": doc.get("nodes") or [],
                "edges": doc.get("edges") or [],
                "summary": summary,
            },
        }

    if kind == "sequence":
        doc = await db.sequence_flow_documents.find_one({"_id": oid, "tenant_id": tenant_oid, "project_id": project_oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Sequence doc not found")
        summary = str(doc.get("summary") or "")
        mermaid = str(doc.get("mermaid") or "").strip()
        md_parts = []
        if summary.strip():
            md_parts.append(summary.strip())
        if mermaid:
            md_parts.append("```mermaid\n" + mermaid + "\n```")
        md = "\n\n".join(md_parts).strip() or "# Sequence Flow\n\n_No summary available._"
        return {
            "doc_id": str(doc.get("_id")),
            "kind": "sequence",
            "project_id": str(project_id),
            "version": int(doc.get("version") or 1),
            "created_at": _iso(doc.get("generated_at") or doc.get("created_at")),
            "content": md,
            "raw": {
                "nodes": doc.get("nodes") or [],
                "edges": doc.get("edges") or [],
                "summary": summary,
                "mermaid": mermaid,
            },
        }

    # architecture
    doc = await db.architecture_diagram_documents.find_one({"_id": oid, "tenant_id": tenant_oid, "project_id": project_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Architecture doc not found")
    summary = str(doc.get("summary") or "")
    mermaid = str(doc.get("mermaid") or "").strip()
    view = str(doc.get("view") or "").strip()
    md_parts = []
    if view:
        md_parts.append(f"### View\n\n{view}")
    if summary.strip():
        md_parts.append(summary.strip())
    if mermaid:
        md_parts.append("```mermaid\n" + mermaid + "\n```")
    md = "\n\n".join(md_parts).strip() or "# Architecture Diagram\n\n_No summary available._"
    return {
        "doc_id": str(doc.get("_id")),
        "kind": "architecture",
        "project_id": str(project_id),
        "version": int(doc.get("version") or 1),
        "created_at": _iso(doc.get("generated_at") or doc.get("created_at")),
        "content": md,
        "raw": {"summary": summary, "mermaid": mermaid, "view": view},
    }
