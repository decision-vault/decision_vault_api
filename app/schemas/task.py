from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# --- SPRINT SCHEMAS ---
class SprintCreate(BaseModel):
    name: str
    description: Optional[str] = None
    start_date: datetime
    end_date: datetime
    project_id: str

class SprintUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

# --- TASK & SUBTASK SCHEMAS ---
class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    role: Optional[str] = "backend_developer"
    priority: Optional[str] = "medium"
    story_points: Optional[int] = 3
    project_id: str
    sprint_id: Optional[str] = None
    parent_id: Optional[str] = None  #  Added to support Subtask association

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    priority: Optional[str] = None
    story_points: Optional[int] = None
    status: Optional[str] = None
    sprint_id: Optional[str] = None
    parent_id: Optional[str] = None  #  Allows moving subtasks between parents

class TaskStatusUpdate(BaseModel):
    status: str

class TaskAssign(BaseModel):
    user_id: str

class CommentCreate(BaseModel):
    message: str