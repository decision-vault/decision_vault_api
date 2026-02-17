from typing import Optional

from pydantic import BaseModel, Field


class ArchitectureDecisions(BaseModel):
    authentication_strategy: Optional[str] = None
    authorization_rbac_model: Optional[str] = None
    data_sync_strategy: Optional[str] = None
    offline_support: Optional[str] = None  # yes/no or description
    currency_handling: Optional[str] = None
    multi_platform_support: Optional[str] = None
    monitoring_and_logging: Optional[str] = None


class TechStack(BaseModel):
    frontend_choice: Optional[str] = None
    backend_choice: Optional[str] = None
    database_choice: Optional[str] = None
    infra_region: Optional[str] = None
    deployment_strategy: Optional[str] = None


class NonFunctional(BaseModel):
    security_requirements: Optional[str] = None
    performance_goals: Optional[str] = None
    compliance_requirements: Optional[str] = None


class Constraints(BaseModel):
    hard_constraints: Optional[list[str]] = None


class RequirementsPartial(BaseModel):
    project_name: Optional[str] = None
    problem_statement: Optional[str] = None
    target_users: Optional[list[str]] = None
    desired_features: Optional[list[str]] = None
    architecture_decisions: Optional[ArchitectureDecisions] = None
    tech_stack: Optional[TechStack] = None
    non_functional: Optional[NonFunctional] = None
    success_metrics: Optional[list[str]] = None
    constraints: Optional[Constraints] = None


class PRDSchema(BaseModel):
    project_name: str
    problem_statement: str
    target_users: list[str]
    desired_features: list[str]
    architecture_decisions: ArchitectureDecisions
    tech_stack: TechStack
    non_functional: NonFunctional
    success_metrics: list[str]
    constraints: Constraints
    out_of_scope: list[str]


class RequirementsStartRequest(BaseModel):
    raw_text: str = Field(..., min_length=10, max_length=12000)


class RequirementsStartResponse(BaseModel):
    intake_id: str
    structured_partial: RequirementsPartial
    missing_fields: list[str]
    low_quality_fields: list[str]
    questions: list[str]


class RequirementsRespondRequest(BaseModel):
    intake_id: str
    answers: dict


class RequirementsRespondResponse(BaseModel):
    structured_partial: RequirementsPartial
    missing_fields: list[str]
    low_quality_fields: list[str]
    questions: list[str]
    ready_for_prd: bool


class RequirementsGenerateRequest(BaseModel):
    intake_id: str


class RequirementsGenerateResponse(BaseModel):
    prd: PRDSchema
