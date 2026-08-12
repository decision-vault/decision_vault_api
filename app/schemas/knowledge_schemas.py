from pydantic import BaseModel, Field
from typing import List, Optional


class SourceDocumentStats(BaseModel):
    document_id: str
    title: str
    chunks: int = 0


class KnowledgeIndexResponse(BaseModel):
    indexed_documents: int = 0
    chunks_created: int = 0
    decisions_extracted: int = 0
    sources: List[SourceDocumentStats] = Field(default_factory=list)


class KnowledgeSource(BaseModel):
    document_id: str
    source_title: str
    chunk_text: str
    score: float = 0.0


class DecisionRecord(BaseModel):
    title: str
    context: str = ""
    alternatives: List[str] = Field(default_factory=list)
    choice: str = ""
    rationale: str = ""
    outcome: str = ""
    source_document_id: str = ""
    source_title: str = ""


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: List[KnowledgeSource] = Field(default_factory=list)
    decisions: List[DecisionRecord] = Field(default_factory=list)


class Citation(BaseModel):
    source_title: str
    document_id: str
    chunk_text: str
    score: float = 0.0
