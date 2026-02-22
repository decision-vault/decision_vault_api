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
