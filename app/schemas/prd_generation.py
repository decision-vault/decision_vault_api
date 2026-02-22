from pydantic import BaseModel


class PRDGenerateRequest(BaseModel):
    title: str
    problem_statement: str
    target_users: str
    features: list[str]
    additional_notes: str | None = None


class PRDGenerateResponse(BaseModel):
    status: str = "ready_for_prd_generation"
    prd_markdown: str
    confidence_score: float
    sections_generated: list[str]


class PRDClarificationResponse(BaseModel):
    status: str = "clarification_required"
    questions: list[str]
    completion_score: float


class PRDStructuredOutput(BaseModel):
    executive_summary: str
    problem_statement: str
    target_users: list[str]
    feature_overview: list[str]
    technical_considerations: list[str]
    risks: list[str]
    success_metrics: list[str]


class PRDClarificationRespondRequest(BaseModel):
    draft: PRDGenerateRequest
    answers: dict[str, str | list[str]]


class CoreSections(BaseModel):
    executive_summary: str
    problem_statement: str
    target_users_personas: list[str]
    objectives_success_metrics: list[str]


class FeatureSections(BaseModel):
    feature_overview: list[str]
    out_of_scope: list[str]
    constraints: list[str]


class ArchitectureSections(BaseModel):
    architecture_decisions: list[str]
    technical_architecture: list[str]
    deployment_strategy: list[str]


class NonFunctionalSections(BaseModel):
    non_functional_requirements: list[str]
    security_compliance: list[str]


class RiskSections(BaseModel):
    scalability_considerations: list[str]
    risks_mitigation: list[str]
    definition_of_done: list[str]


class PRDMultiStepResponse(BaseModel):
    status: str = "success"
    pages_estimated: int = 5
    sections_generated: list[str]
    total_tokens_used: int
    prd_markdown: str
