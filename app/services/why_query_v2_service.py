from __future__ import annotations

import re
from typing import Iterable

from bson import ObjectId

from app.db.mongo import get_db
from app.services.decision_embedding_service import generate_embedding, search_similar_decisions


PROPOSAL_KEYWORDS = [
    "vs",
    "instead of",
    "considering",
    "comparing",
    "chosen",
    "rather than",
    "switch to",
]


def _extract_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9_\-]+", query.lower())
    return [t for t in terms if len(t) > 2]


def _proposal_detected(query: str) -> bool:
    lowered = query.lower()
    return any(k in lowered for k in PROPOSAL_KEYWORDS)


def _suggested_template(query: str) -> dict:
    return {
        "title": query.strip(),
        "statement": "",
        "context": "",
        "alternatives": [],
        "risks": [],
        "status": "proposed",
    }


def _confidence_from_similarity(max_similarity: float | None) -> str:
    if max_similarity is None:
        return "low"
    if max_similarity >= 0.8:
        return "high"
    if max_similarity >= 0.6:
        return "medium"
    return "low"


async def _keyword_search(tenant_id: str, project_id: str, query: str, limit: int) -> list[dict]:
    db = get_db()
    terms = _extract_terms(query)
    if not terms:
        return []
    regex = "|".join(re.escape(t) for t in terms)
    query_filter = {
        "tenant_id": ObjectId(tenant_id),
        "project_id": ObjectId(project_id),
        "$or": [
            {"title": {"$regex": regex, "$options": "i"}},
            {"statement": {"$regex": regex, "$options": "i"}},
            {"context": {"$regex": regex, "$options": "i"}},
        ],
    }
    cursor = db.decisions.find(query_filter).sort("timestamp", -1).limit(limit)
    results = []
    async for doc in cursor:
        results.append(
            {
                "id": str(doc.get("_id")),
                "title": doc.get("title") or "",
                "excerpt": (doc.get("statement") or doc.get("context") or "")[:240],
                "similarity": 1.0,
            }
        )
    return results


async def run_why_query_v2(
    tenant_id: str,
    project_id: str,
    query: str,
    limit: int = 5,
    threshold: float = 0.6,
) -> dict:
    related_decisions: list[dict] = []

    exact_matches = await _keyword_search(tenant_id, project_id, query, limit)
    related_decisions.extend(exact_matches)

    query_embedding = generate_embedding(query)
    semantic_matches = await search_similar_decisions(
        tenant_id=tenant_id,
        project_id=project_id,
        query_embedding=query_embedding,
        threshold=threshold,
        limit=limit,
    )

    existing_ids = {d["id"] for d in related_decisions}
    for match in semantic_matches:
        if match["id"] not in existing_ids:
            related_decisions.append(match)

    max_similarity = None
    if related_decisions:
        max_similarity = max(d["similarity"] for d in related_decisions)

    answer = "No decision found. Create one?"
    if related_decisions:
        answer = "Found related decisions based on your query."

    response: dict = {
        "answer": answer,
        "related_decisions": related_decisions,
        "confidence": _confidence_from_similarity(max_similarity),
    }

    strong_match = max_similarity is not None and max_similarity >= 0.75
    if _proposal_detected(query) and not strong_match:
        response["suggestion"] = "This looks like a new proposal. Consider capturing it as a decision."
        response["suggested_decision_template"] = _suggested_template(query)

    return response
