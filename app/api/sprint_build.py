"""Proxy endpoints for sprint build agent — forwards to LangGraph agent service."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import logging

from app.core.config import settings

logger = logging.getLogger("api.sprint_build")

router = APIRouter(prefix="/api/sprint-build", tags=["Sprint Build"])


class SprintBuildPayload(BaseModel):
    sprint_id: Any
    project_id: Any
    tasks: List[Dict[str, Any]]
    prd_context: Optional[str] = ""
    domain_context: Optional[Dict[str, Any]] = {}
    project_dir: Optional[str] = ""

class PermissionAction(BaseModel):
    sprint_id: str
    task_id: str


@router.get("/active/{project_id}")
async def get_active_build(project_id: str):
    """Proxy: Find active build for project → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/active/{project_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(agent_url)
            if resp.status_code != 200:
                return {"status": "not_found"}
            return resp.json()
    except httpx.ConnectError:
        return {"status": "not_found"}


@router.post("/start")
async def start_sprint_build(payload: SprintBuildPayload):
    """Proxy: Start sprint build → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/start"
    logger.info(f"Proxying sprint build start to {agent_url}")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(agent_url, json=payload.model_dump())
            if resp.status_code != 200:
                logger.error(f"Agent returned {resp.status_code}: {resp.text[:200]}")
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            return resp.json()
    except httpx.ConnectError as e:
        logger.error(f"Cannot reach agent at {settings.langgraph_url}: {e}")
        raise HTTPException(status_code=502, detail=f"Agent service unreachable at {settings.langgraph_url}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Agent service timed out")


@router.get("/{sprint_id}/status")
async def sprint_build_status(sprint_id: str):
    """Proxy: Get sprint build status → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/{sprint_id}/status"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(agent_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Agent service unreachable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Agent service timed out")


@router.get("/{sprint_id}/progress")
async def sprint_build_progress(sprint_id: str):
    """Proxy SSE: Stream sprint build progress → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/{sprint_id}/progress"
    logger.info(f"Proxying SSE stream from {agent_url}")

    async def proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=360.0) as client:
                async with client.stream("GET", agent_url) as resp:
                    logger.info(f"SSE proxy connected to agent, status={resp.status_code}")
                    async for chunk in resp.aiter_text():
                        yield chunk
        except httpx.ConnectError as e:
            logger.error(f"SSE proxy connect error: {e}")
            yield f"data: {{\"type\": \"error\", \"message\": \"Agent service unreachable\"}}\n\n"
        except Exception as e:
            logger.error(f"SSE proxy error: {e}")
            yield f"data: {{\"type\": \"error\", \"message\": \"{str(e)[:200]}\"}}\n\n"

    return StreamingResponse(
        proxy_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Permission proxy endpoints ─────────────────────────────────────

@router.get("/permissions/pending")
async def get_pending_permissions():
    """Proxy: Get pending permission requests → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/permissions/pending"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(agent_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Agent service unreachable")


@router.post("/permissions/approve")
async def approve_permission(payload: PermissionAction):
    """Proxy: Approve permission → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/permissions/approve"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(agent_url, json=payload.model_dump())
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Agent service unreachable")


@router.post("/permissions/deny")
async def deny_permission(payload: PermissionAction):
    """Proxy: Deny permission → agent service."""
    agent_url = f"{settings.langgraph_url}/workflow/sprint/permissions/deny"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(agent_url, json=payload.model_dump())
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Agent service unreachable")
