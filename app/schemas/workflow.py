from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class WorkflowBase(BaseModel):
    project_id: str
    nodes: List[Dict[str, Any]] = Field(default=[])
    edges: List[Dict[str, Any]] = Field(default=[])

class WorkflowCreate(WorkflowBase):
    pass

class WorkflowUpdate(BaseModel):
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None

class WorkflowResponse(WorkflowBase):
    id: str
    tenant_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
