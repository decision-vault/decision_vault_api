from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TypedDict

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import settings
from bson import ObjectId

from app.db.mongo import get_db
from app.schemas.why_query import WhyAnswerDraft, WhyDecisionItem, WhyQueryResponse


@dataclass
class DecisionRecord:
    decision_id: str
    title: str
    statement: str
    context: str | None
    alternatives: str | None
    risks: str | None
    source_url: str
    decided_at: date | None


class WhyQueryState(TypedDict):
    query: str
    tenant_id: str
    project_id: str
    decisions: list[DecisionRecord]
    answer: str | None
    cited_decisions: list[str]
    confidence: str


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().split())


async def validate_query(state: WhyQueryState) -> WhyQueryState:
    query = _normalize_query(state["query"])
    if not query or len(query) < 3:
        raise ValueError("Query is too short")
    return {**state, "query": query}


async def retrieve_decisions(state: WhyQueryState) -> WhyQueryState:
    query = state["query"]
    tenant_id = state["tenant_id"]
    project_id = state["project_id"]
    if not ObjectId.is_valid(tenant_id) or not ObjectId.is_valid(project_id):
        raise ValueError("Invalid tenant_id or project_id")
    db = get_db()
    keywords = [term for term in query.split(" ") if term]
    base_or = [
        {"title": {"$regex": query, "$options": "i"}},
        {"statement": {"$regex": query, "$options": "i"}},
        {"context": {"$regex": query, "$options": "i"}},
        {"alternatives": {"$regex": query, "$options": "i"}},
        {"risks": {"$regex": query, "$options": "i"}},
    ]
    and_terms = []
    for term in keywords:
        and_terms.append(
            {
                "$or": [
                    {"title": {"$regex": term, "$options": "i"}},
                    {"statement": {"$regex": term, "$options": "i"}},
                    {"context": {"$regex": term, "$options": "i"}},
                    {"alternatives": {"$regex": term, "$options": "i"}},
                    {"risks": {"$regex": term, "$options": "i"}},
                ]
            }
        )
    query_filter: dict = {
        "tenant_id": ObjectId(tenant_id),
        "project_id": ObjectId(project_id),
        "$or": base_or,
    }
    if and_terms:
        query_filter["$and"] = and_terms
    cursor = (
        db.decisions.find(query_filter)
        .sort("decided_at", -1)
        .limit(10)
    )
    decisions = []
    async for row in cursor:
        decided_at = row.get("decided_at") or row.get("timestamp") or row.get("created_at")
        if hasattr(decided_at, "date"):
            decided_date = decided_at.date()
        else:
            decided_date = decided_at
        decisions.append(
            DecisionRecord(
                decision_id=str(row.get("_id")),
                title=row.get("title") or "",
                statement=row.get("statement") or "",
                context=row.get("context"),
                alternatives=row.get("alternatives"),
                risks=row.get("risks"),
                source_url=row.get("source_url") or "",
                decided_at=decided_date if decided_at else None,
            )
        )
    return {**state, "decisions": decisions}


async def evidence_filter(state: WhyQueryState) -> WhyQueryState:
    if not state.get("decisions"):
        return {**state, "answer": None, "confidence": "low"}
    return state


def _build_prompt(parser: BaseOutputParser) -> ChatPromptTemplate:
    system = (
        "You are a decision analyst. Answer ONLY using the provided decisions.\n"
        "If the answer is not explicitly present, say you don't know.\n"
        "Cite decisions with decision_id and source_url.\n"
    )
    user = (
        "Question: {query}\n\n"
        "Decisions:\n{decisions}\n\n"
        "Return JSON with answer_text, cited_decisions, confidence.\n"
        "{format_instructions}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user),
        ]
    )


async def answer_composer(state: WhyQueryState) -> WhyQueryState:
    if not state.get("decisions"):
        return {**state, "answer": None, "cited_decisions": [], "confidence": "low"}

    parser = PydanticOutputParser(pydantic_object=WhyAnswerDraft)
    prompt = _build_prompt(parser)
    if not settings.llm_api_key:
        raise RuntimeError("LLM API key not configured")
    if settings.llm_provider == "openai":
        if not settings.llm_model:
            raise RuntimeError("LLM model not configured")
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.llm_api_key,
        )
    elif settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError("Gemini dependencies not installed") from exc
        if not settings.llm_model:
            raise RuntimeError("LLM model not configured")
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.llm_api_key,
        )
    else:
        raise RuntimeError("Unsupported LLM provider")

    decision_text = "\n".join(
        [
            f"- decision_id: {d.decision_id}\n"
            f"  title: {d.title}\n"
            f"  statement: {d.statement}\n"
            f"  context: {d.context}\n"
            f"  alternatives: {d.alternatives}\n"
            f"  risks: {d.risks}\n"
            f"  source_url: {d.source_url}\n"
            for d in state["decisions"]
        ]
    )

    chain = prompt | llm | parser
    result: WhyAnswerDraft = await chain.ainvoke(
        {
            "query": state["query"],
            "decisions": decision_text,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    retrieved_ids = {d.decision_id for d in state["decisions"]}
    cited_ids = [decision_id for decision_id in result.cited_decisions if decision_id in retrieved_ids]
    if not cited_ids:
        return {**state, "answer": None, "cited_decisions": [], "confidence": "low"}

    confidence = result.confidence
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        **state,
        "answer": result.answer_text,
        "cited_decisions": cited_ids,
        "confidence": confidence,
    }


async def response_formatter(state: WhyQueryState) -> WhyQueryResponse:
    if not state.get("decisions") or not state.get("answer"):
        return WhyQueryResponse(answer="No decision found. Create one?", decisions=[], confidence="low")

    decision_map = {d.decision_id: d for d in state["decisions"]}
    decisions = []
    for decision_id in state["cited_decisions"]:
        record = decision_map.get(decision_id)
        if not record:
            continue
        decisions.append(
            WhyDecisionItem(
                decision_id=record.decision_id,
                title=record.title,
                date=record.decided_at,
                source_url=record.source_url,
            )
        )

    if not decisions:
        return WhyQueryResponse(answer="No decision found. Create one?", decisions=[], confidence="low")

    return WhyQueryResponse(
        answer=state["answer"],
        decisions=decisions,
        confidence=state["confidence"],
    )


def build_graph():
    # LangGraph gives us a deterministic, explicit control flow and guarantees we stop
    # when evidence is missing. This avoids a single LLM call that might hallucinate.
    graph = StateGraph(WhyQueryState)
    graph.add_node("ValidateQuery", validate_query)
    graph.add_node("RetrieveDecisions", retrieve_decisions)
    graph.add_node("EvidenceFilter", evidence_filter)
    graph.add_node("AnswerComposer", answer_composer)
    graph.add_node("ResponseFormatter", response_formatter)

    graph.set_entry_point("ValidateQuery")
    graph.add_edge("ValidateQuery", "RetrieveDecisions")
    graph.add_edge("RetrieveDecisions", "EvidenceFilter")
    graph.add_edge("EvidenceFilter", "AnswerComposer")
    graph.add_edge("AnswerComposer", "ResponseFormatter")
    graph.add_edge("ResponseFormatter", END)
    return graph.compile()


async def run_why_query(tenant_id: str, project_id: str, query: str) -> WhyQueryResponse:
    graph = build_graph()
    state: WhyQueryState = {
        "query": query,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "decisions": [],
        "answer": None,
        "cited_decisions": [],
        "confidence": "low",
    }
    result = await graph.ainvoke(state)
    if isinstance(result, WhyQueryResponse):
        return result
    return WhyQueryResponse(**result)


async def _vector_search(*_args, **_kwargs) -> list[DecisionRecord]:
    # Stage 1: Mongo-only retrieval. Vector search can be reintroduced once pgvector is wired.
    return []
