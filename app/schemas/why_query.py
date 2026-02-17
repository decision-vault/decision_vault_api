from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class WhyDecisionItem(BaseModel):
    decision_id: str
    title: str
    date: Optional[date] = None
    source_url: str


class WhyAnswerDraft(BaseModel):
    answer_text: str
    cited_decisions: list[str]
    confidence: str = Field(..., pattern="^(high|medium|low)$")


class WhyQueryResponse(BaseModel):
    answer: str
    decisions: list[WhyDecisionItem]
    confidence: str = Field(..., pattern="^(high|medium|low)$")


class WhyQueryExample(BaseModel):
    query: str
    response: WhyQueryResponse


class WhyRelatedDecisionItem(BaseModel):
    id: str
    title: str
    excerpt: str
    similarity: float


class WhyQueryV2Response(BaseModel):
    answer: str
    related_decisions: list[WhyRelatedDecisionItem]
    suggestion: str | None = None
    suggested_decision_template: dict | None = None
    confidence: str = Field(..., pattern="^(high|medium|low)$")
