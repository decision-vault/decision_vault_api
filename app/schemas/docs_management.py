from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# --- SYSTEM TIMELINE SNAPSHOT SCHEMAS ---
class ChatVersionLog(BaseModel):
    timestamp: str = Field(..., description="ISO string timestamp tracking the exact moment of modification")
    agent_prompt_or_chat: str = Field(..., description="The context prompt or user command triggering this revision snapshot")
    saved_snapshot_body: str = Field(..., description="The complete HTML/Quill content snapshot state captured at this exact mark")
    is_plan_card: Optional[bool] = Field(default=False, description="Flag indicating if this message represents a generated project plan card")

# --- DOCUMENT SCHEMAS ---
class DocumentBase(BaseModel):
    title: str = Field(..., max_length=150, description="Title of the workspace document")
    body: Optional[str] = Field("", description="HTML raw text markup string from the Quill rich text editor engine")

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: str
    workspace_id: str
    updated_at: datetime
    chat_history: List[ChatVersionLog] = Field(default=[], description="Append-only collection tracking chat history and code revisions")

    class Config:
        populate_by_name = True
        from_attributes = True

# --- WORKSPACE SCHEMAS ---
class WorkspaceCreate(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the top-level organizational space layer")

class WorkspaceResponse(BaseModel):
    id: str
    name: str
    project_id: Optional[str] = None
    documents: List[DocumentResponse] = []
    created_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
        
class WorkspaceDeleteResponse(BaseModel):
    success: bool
    message: str