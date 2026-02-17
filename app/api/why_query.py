from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.middleware.guard import withGuard
from app.schemas.why_query import WhyQueryResponse, WhyQueryV2Response
from app.services.why_query_service import run_why_query
from app.services.why_query_v2_service import run_why_query_v2


router = APIRouter(prefix="/api/projects", tags=["why-query"])

EXAMPLE_QUERY = "Why did we choose PostgreSQL over MongoDB?"
EXAMPLE_RESPONSE = {
    "answer": "We chose PostgreSQL for stronger relational integrity and SQL reporting needs.",
    "decisions": [
        {
            "decision_id": "9d7e1a2b-1234-4eaf-9e1b-6c3f1e2d4a55",
            "title": "Database selection",
            "date": "2025-08-12",
            "source_url": "https://slack.com/archives/ABC/p123",
        }
    ],
    "confidence": "medium",
}


@router.get("/{project_id}/why", response_model=WhyQueryResponse)
async def why_query(
    project_id: str,
    request: Request,
    q: str = Query(..., max_length=300),
    _guard=Depends(withGuard(feature="search", projectRole="viewer")),
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    try:
        result = await run_why_query(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            query=q,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.get("/{project_id}/why/v2", response_model=WhyQueryV2Response)
async def why_query_v2(
    project_id: str,
    request: Request,
    q: str = Query(..., max_length=300),
    _guard=Depends(withGuard(feature="search", projectRole="viewer")),
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    result = await run_why_query_v2(
        tenant_id=request.state.tenant_id,
        project_id=project_id,
        query=q,
    )
    return result
