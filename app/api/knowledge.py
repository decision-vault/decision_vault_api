from fastapi import APIRouter, Depends, HTTPException, Query

from app.middleware.guard import withGuard
from app.middleware.tenant import resolve_tenant
from app.schemas.knowledge_schemas import (
    DecisionRecord,
    KnowledgeIndexResponse,
    KnowledgeSearchResponse,
)
from app.services import knowledge_service

router = APIRouter(prefix="/api/projects", tags=["Knowledge Base"])


@router.post("/{project_id}/knowledge/index", response_model=KnowledgeIndexResponse)
async def index_project_knowledge(
    project_id: str,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
    tenant_id: str = Depends(resolve_tenant),
):
    """Chunk every document in the project's workspaces into the KB. Idempotent."""
    result = await knowledge_service.index_project(tenant_id, project_id)
    return result


@router.post("/{project_id}/knowledge/decisions/extract", response_model=KnowledgeIndexResponse)
async def extract_project_decisions(
    project_id: str,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
    tenant_id: str = Depends(resolve_tenant),
):
    """Derive the decision layer: extract decision records from indexed documents."""
    await knowledge_service.index_project(tenant_id, project_id)
    extracted = await knowledge_service.extract_decisions_for_project(tenant_id, project_id)
    stats = await knowledge_service.index_project(tenant_id, project_id)
    stats["decisions_extracted"] = extracted
    return stats


@router.get("/{project_id}/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_project_knowledge(
    project_id: str,
    q: str = Query(..., min_length=1),
    top_k: int = Query(4, ge=1, le=10),
    user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
    tenant_id: str = Depends(resolve_tenant),
):
    return await knowledge_service.search(tenant_id, project_id, q, top_k=top_k)


@router.get("/{project_id}/knowledge/decisions", response_model=list[DecisionRecord])
async def list_project_decisions(
    project_id: str,
    user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
    tenant_id: str = Depends(resolve_tenant),
):
    from app.db.mongo import get_db

    db = get_db()
    cursor = db.decision_records.find(
        {"tenant_id": tenant_id, "project_id": project_id}
    ).sort("updated_at", -1)
    records = await cursor.to_list(length=200)
    return [
        {
            "title": r.get("title", ""),
            "context": r.get("context", ""),
            "alternatives": r.get("alternatives", []),
            "choice": r.get("choice", ""),
            "rationale": r.get("rationale", ""),
            "outcome": r.get("outcome", ""),
            "source_document_id": r.get("document_id", ""),
            "source_title": r.get("source_title", ""),
        }
        for r in records
    ]
