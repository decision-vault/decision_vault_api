from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class AIWorkflowTaskBase(BaseModel):
    title: str
    description: str
    phase: str
    priority: str = "medium"
    status: str = "pending"
    story_points: int = 3
    estimated_hours: float = 24.0
    assigned_agent: str
    dependencies: List[Dict[str, Any]] = Field(default=[])
    acceptance_criteria: List[str] = Field(default=[])
    artifacts: List[str] = Field(default=[])
    assigned_human_id: Optional[str] = None

class AIWorkflowPhaseBase(BaseModel):
    name: str
    order: int
    status: str = "pending"
    tasks: List[AIWorkflowTaskBase] = Field(default=[])

class AIWorkflowStatistics(BaseModel):
    total_tasks: int = 0
    completed_tasks: int = 0
    blocked_tasks: int = 0
    total_story_points: int = 0
    estimated_sprint_count: int = 1
    velocity: int = 30
    progress: float = 0.0

class AIWorkflowResponse(BaseModel):
    id: str
    project_id: str
    tenant_id: str
    status: str = "active"
    phases: List[AIWorkflowPhaseBase] = Field(default=[])
    statistics: AIWorkflowStatistics = Field(default_factory=AIWorkflowStatistics)
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
