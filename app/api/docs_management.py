from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from app.schemas.docs_management import (
    WorkspaceCreate, WorkspaceResponse, DocumentCreate, DocumentUpdate, DocumentResponse, WorkspaceDeleteResponse
)
from app.services.docs_management_service import DocsManagementService
from app.middleware.tenant import resolve_tenant
from app.utils.serialize import serialize_doc

router = APIRouter(prefix="/api/docs-management", tags=["Document Management Workspace"])

@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_single_document(document_id: str):
    """
    Fetches a single document configuration directly by its ID.
    Enables background LangGraph nodes to read and parse the text before applying inline updates.
    """
    doc = await DocsManagementService.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Target document context configuration lane missing"
        )
    
    serialized = serialize_doc(doc)
    serialized["id"] = str(serialized["_id"])
    if "chat_history" not in serialized:
        serialized["chat_history"] = []
    return serialized

@router.get("/workspaces", response_model=List[WorkspaceResponse])
async def list_workspaces(tenant_id: str = Depends(resolve_tenant)):
    workspaces = await DocsManagementService.get_all_workspaces(tenant_id)
    serialized = []
    for ws in workspaces:
        ws_doc = serialize_doc(ws)
        ws_doc["id"] = str(ws_doc["_id"])
        if "documents" in ws_doc:
            for d in ws_doc["documents"]:
                d["id"] = str(d["_id"])
                if "chat_history" not in d:
                    d["chat_history"] = []
        serialized.append(ws_doc)
    return serialized

@router.post("/workspaces", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(payload: WorkspaceCreate, tenant_id: str = Depends(resolve_tenant)):
    ws = await DocsManagementService.create_workspace(tenant_id, payload.model_dump())
    serialized = serialize_doc(ws)
    serialized["id"] = str(serialized["_id"])
    serialized["documents"] = []
    return serialized

@router.post("/workspaces/{workspace_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(workspace_id: str, payload: DocumentCreate, tenant_id: str = Depends(resolve_tenant)):
    doc = await DocsManagementService.create_document(tenant_id, workspace_id, payload.model_dump())
    serialized = serialize_doc(doc)
    serialized["id"] = str(serialized["_id"])
    serialized["chat_history"] = []
    return serialized

@router.patch("/documents/{document_id}", response_model=DocumentResponse)
async def sync_quill_editor_changes(
    document_id: str, 
    payload: DocumentUpdate, 
    agent_chat_msg: Optional[str] = None
):
    updated = await DocsManagementService.update_document(
        document_id=document_id, 
        payload=payload.model_dump(exclude_none=True),
        agent_chat_msg=agent_chat_msg
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Target document configuration layer missing")
    serialized = serialize_doc(updated)
    serialized["id"] = str(serialized["_id"])
    return serialized

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    success = await DocsManagementService.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Target tracking object context missing")
    return

@router.delete("/workspaces/{workspace_id}", response_model=WorkspaceDeleteResponse)
async def delete_workspace(workspace_id: str, tenant_id: str = Depends(resolve_tenant)):
    # This now runs the updated service with full document cascading hooks!
    success = await DocsManagementService.delete_workspace(tenant_id, workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found or unauthorized.")
    return {"success": True, "message": "Workspace and all associated PRD documents purged successfully."}