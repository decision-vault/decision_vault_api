from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
import logging
from bson import ObjectId
from app.schemas.canvas import CanvasCreate, CanvasUpdate, CanvasResponse
from app.services.canvas_service import CanvasService
from app.services.docs_management_service import DocsManagementService
from app.middleware.guard import withGuard
from app.core.config import settings
from app.db.mongo import get_db
from app.utils.serialize import serialize_doc

logger = logging.getLogger("decisionvault.canvases")

router = APIRouter(prefix="/api/canvases", tags=["Creative UI Architecture Canvas"])


class GenerateCanvasRequest(BaseModel):
    project_id: str
    document_id: str


@router.post("/generate", response_model=CanvasResponse)
async def generate_canvas(
    project_id: str,
    payload: GenerateCanvasRequest,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    tenant_id = request.state.tenant_id
    db = get_db()

    doc = await DocsManagementService.get_document_by_id(payload.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    prd_body = doc.get("body", "")

    project = await db.projects.find_one({"_id": ObjectId(payload.project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    product_name = project.get("name", "New Project")

    agent_url = f"{settings.langgraph_url}/workflow/ui-builder/generate"
    ui_payload = {"product_name": product_name, "prd_body": prd_body}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(agent_url, json=ui_payload, timeout=120.0)
            if resp.status_code != 200:
                logger.error(f"Agent returned status {resp.status_code}: {resp.text[:500]}")
                raise HTTPException(status_code=500, detail=f"UI builder agent failed: {resp.text}")
            layout_data = resp.json()
            logger.info(f"Agent response keys: {list(layout_data.keys()) if isinstance(layout_data, dict) else type(layout_data)}")
            logger.info(f"Pages count: {len(layout_data.get('pages', [])) if isinstance(layout_data, dict) else 'N/A'}")
        except httpx.TimeoutException:
            logger.error("Agent timed out")
            raise HTTPException(status_code=504, detail="UI builder agent timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to reach agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to reach UI builder agent: {str(e)}")

    if not layout_data:
        logger.error("Agent returned empty response")
        raise HTTPException(status_code=500, detail="UI builder agent returned empty response")

    if "pages" in layout_data:
        layout = layout_data
        logger.info(f"Using 'pages' key directly: {len(layout.get('pages', []))} pages")
    elif "layout_json" in layout_data:
        layout = layout_data["layout_json"]
        logger.info(f"Using 'layout_json' key: {len(layout.get('pages', []))} pages")
    else:
        layout = layout_data
        logger.info(f"Using layout_data directly: {len(layout.get('pages', []))} pages")

    has_content = len(layout.get("pages", [])) > 0 or len(layout.get("nodes", [])) > 0
    if not has_content:
        logger.error(f"No pages or nodes found. Layout keys: {list(layout.keys()) if isinstance(layout, dict) else type(layout)}")
        raise HTTPException(status_code=500, detail="UI builder returned layout with no pages")
    else:
        logger.info(f"Layout has content: {len(layout.get('pages', []))} pages, {len(layout.get('nodes', []))} nodes")

    existing = await CanvasService.get_canvas_by_project(payload.project_id)
    if existing:
        await CanvasService.update_canvas(payload.project_id, {"layout_json": layout})
    else:
        await CanvasService.create_canvas(tenant_id=tenant_id, payload={"project_id": payload.project_id, "layout_json": layout})

    canvas = await CanvasService.get_canvas_by_project(payload.project_id)
    serialized = serialize_doc(canvas)
    serialized["id"] = str(serialized["_id"])
    return serialized


@router.post("/generate-stream")
async def generate_canvas_stream(
    project_id: str,
    payload: GenerateCanvasRequest,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    tenant_id = request.state.tenant_id
    db = get_db()

    doc = await DocsManagementService.get_document_by_id(payload.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    prd_body = doc.get("body", "")

    project = await db.projects.find_one({"_id": ObjectId(payload.project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    product_name = project.get("name", "New Project")

    agent_url = f"{settings.langgraph_url}/workflow/ui-builder/generate"
    ui_payload = {"product_name": product_name, "prd_body": prd_body}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(agent_url, json=ui_payload, timeout=120.0)
            if resp.status_code != 200:
                logger.error(f"Agent returned status {resp.status_code}: {resp.text[:500]}")
                raise HTTPException(status_code=500, detail=f"UI builder agent failed: {resp.text}")
            layout_data = resp.json()
            logger.info(f"Stream endpoint - Agent response keys: {list(layout_data.keys()) if isinstance(layout_data, dict) else type(layout_data)}")
            logger.info(f"Stream endpoint - Pages count: {len(layout_data.get('pages', [])) if isinstance(layout_data, dict) else 'N/A'}")
        except httpx.TimeoutException:
            logger.error("Stream endpoint - Agent timed out")
            raise HTTPException(status_code=504, detail="UI builder agent timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Stream endpoint - Failed to reach agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to reach UI builder agent: {str(e)}")

    if not layout_data:
        logger.error("Stream endpoint - Agent returned empty response")
        raise HTTPException(status_code=500, detail="UI builder agent returned empty response")

    if "pages" in layout_data:
        layout = layout_data
        logger.info(f"Stream endpoint - Using 'pages' key directly: {len(layout.get('pages', []))} pages")
    elif "layout_json" in layout_data:
        layout = layout_data["layout_json"]
        logger.info(f"Stream endpoint - Using 'layout_json' key: {len(layout.get('pages', []))} pages")
    else:
        layout = layout_data
        logger.info(f"Stream endpoint - Using layout_data directly: {len(layout.get('pages', []))} pages")

    pages = layout.get("pages", [])
    edges = layout.get("edges", [])
    if not pages:
        logger.error("Stream endpoint - No pages found")
        raise HTTPException(status_code=500, detail="UI builder returned layout with no pages")
    else:
        logger.info(f"Stream endpoint - Streaming {len(pages)} pages with {len(edges)} edges")

    existing = await CanvasService.get_canvas_by_project(payload.project_id)
    if existing:
        await CanvasService.update_canvas(payload.project_id, {"layout_json": layout})
    else:
        await CanvasService.create_canvas(tenant_id=tenant_id, payload={"project_id": payload.project_id, "layout_json": layout})

    async def event_stream():
        for i, page in enumerate(pages):
            event_data = json.dumps({"page": page, "index": i, "total": len(pages)})
            yield f"event: page\ndata: {event_data}\n\n"
            await asyncio.sleep(0.3)
        complete_data = json.dumps({"total": len(pages), "edges": edges})
        yield f"event: complete\ndata: {complete_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@router.post("", response_model=CanvasResponse)
async def create_canvas(
    payload: CanvasCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    tenant_id = request.state.tenant_id
    existing = await CanvasService.get_canvas_by_project(payload.project_id)
    if existing:
        raise HTTPException(status_code=400, detail="Canvas for this project already exists.")
    
    canvas = await CanvasService.create_canvas(tenant_id=tenant_id, payload=payload.model_dump())
    serialized = serialize_doc(canvas)
    serialized["id"] = str(serialized["_id"])
    return serialized

@router.get("", response_model=CanvasResponse)
async def get_canvas(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="view_decision", projectRole="viewer"))
):
    canvas = await CanvasService.get_canvas_by_project(project_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
    serialized = serialize_doc(canvas)
    serialized["id"] = str(serialized["_id"])
    return serialized

@router.put("", response_model=CanvasResponse)
async def update_canvas(
    project_id: str,
    payload: CanvasUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    canvas = await CanvasService.update_canvas(project_id, payload.model_dump(exclude_none=True))
    if not canvas:
        raise HTTPException(status_code=404, detail="Canvas not found")
    serialized = serialize_doc(canvas)
    serialized["id"] = str(serialized["_id"])
    return serialized
