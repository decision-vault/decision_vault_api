from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class CanvasBase(BaseModel):
    project_id: str
    layout_json: Dict[str, Any] = Field(default={})

class CanvasCreate(CanvasBase):
    pass

class CanvasUpdate(BaseModel):
    layout_json: Optional[Dict[str, Any]] = None

class CanvasResponse(CanvasBase):
    id: str
    tenant_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
