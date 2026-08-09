import traceback
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
import httpx
from app.schemas.prd_generator_schemas import InteractivePRDRequest, InteractivePRDResponse, JobStatusResponse
from app.services.docs_management_service import DocsManagementService
from app.middleware.tenant import resolve_tenant
from app.core.config import settings
from app.db.mongo import get_db

logger = logging.getLogger("decisionvault.prd_generator")
router = APIRouter(prefix="/api/prd-generator", tags=["AI PRD Automation Engine"])

from bson import ObjectId
from datetime import datetime

ACTIVE_PROCESSING_JOBS = {}

async def _set_job(job_id: str, data: dict) -> None:
    """Merge updates into a job. Backed by MongoDB (survives serverless) with an
    in-memory cache for fast local reads."""
    ACTIVE_PROCESSING_JOBS[job_id] = {**ACTIVE_PROCESSING_JOBS.get(job_id, {}), **data}
    try:
        db = get_db()
        await db.prd_generation_jobs.update_one(
            {"job_id": job_id},
            {"$set": {**ACTIVE_PROCESSING_JOBS[job_id], "updated_at": datetime.utcnow()}},
            upsert=True,
        )
    except Exception as persist_err:
        logger.warning(f"Failed to persist job {job_id}: {persist_err}")


async def _get_job(job_id: str) -> dict | None:
    if job_id in ACTIVE_PROCESSING_JOBS:
        return ACTIVE_PROCESSING_JOBS[job_id]
    try:
        db = get_db()
        doc = await db.prd_generation_jobs.find_one({"job_id": job_id})
        if doc:
            doc.pop("_id", None)
            return doc
    except Exception as read_err:
        logger.warning(f"Failed to read job {job_id} from MongoDB: {read_err}")
    return None

async def execute_async_langgraph_pipeline(job_id: str, tenant_id: str, payload: InteractivePRDRequest, doc_id: str):
    langgraph_url = f"{settings.langgraph_url}/workflow/prd/interactive-step"
    db = get_db()
    
    messages_payload = []
    
    # 1. If in Chat Mode, retrieve current document specifications to supply as QA context
    if payload.mode == "chat":
        try:
            prd_doc = await db.documents.find_one({"_id": ObjectId(doc_id)})
            doc_body = prd_doc.get("body", "") if prd_doc else ""
            if doc_body:
                messages_payload.append({
                    "role": "system",
                    "content": f"The current active Product Requirements Document specifications are:\n\n{doc_body}"
                })
        except Exception as doc_err:
            logger.error(f"Failed to fetch current doc context for chat QA: {str(doc_err)}")

    for m in payload.messages:
        messages_payload.append(m.model_dump())

    micro_payload = {
        "tenant_id": tenant_id,
        "document_id": doc_id,
        "product_name": payload.product_name,
        "messages": messages_payload,
        "target_audience": payload.target_audience,
        "tech_stack": payload.tech_stack_focus,
        "mode": payload.mode or "chat",
        "current_feature": payload.current_feature or 1
    }

    # Simulate dynamic worker step mapping update lines during handshake initialization phases
    await _set_job(job_id, {"explainability": "Parsing user prompt criteria arrays..."})

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(langgraph_url, json=micro_payload, timeout=None)
            
            if response.status_code != 200:
                await _set_job(job_id, {"status": "failed", "explainability": "Crashed downstream."})
                return
            
            data = response.json()
            ai_text_reply = data.get("response", "")
            dynamic_summary = data.get("change_summary", "Refined functional specification document blocks.")
            is_complete_val = data.get("is_complete", False)

            if payload.mode == "plan":
                await _set_job(job_id, {"explainability": "Committing section mutations to core database models..."})

                await DocsManagementService.update_document(
                    document_id=doc_id,
                    payload={"body": ai_text_reply}, 
                    agent_chat_msg=f"User: {payload.messages[-1].content if payload.messages else ''} | Agent: {ai_text_reply}"
                )

                # Automatically trigger compiling the implementation workflow (task flows, nodes, edges) from the new PRD!
                if payload.project_id:
                    try:
                        await _set_job(job_id, {"explainability": "Compiling project task flows, sprint milestones..."})
                        from app.services.workflow_generator_service import WorkflowGeneratorService
                        await WorkflowGeneratorService.generate_workflow(tenant_id, payload.project_id)
                    except Exception as wf_err:
                        logger.error(f"Failed to generate workflow from PRD: {str(wf_err)}")

                    # Automatically generate the UI canvas mockup wireframe using the updated PRD specifications context!
                    try:
                        await _set_job(job_id, {"explainability": "Designing front-end UI mockups and canvas..."})
                        agent_ui_url = f"{settings.langgraph_url}/workflow/ui-builder/generate"
                        ui_payload = {
                            "product_name": payload.product_name,
                            "prd_body": ai_text_reply
                        }
                        
                        async with httpx.AsyncClient() as client_ui:
                            ui_response = await client_ui.post(
                                agent_ui_url,
                                json=ui_payload,
                                timeout=120.0  # LLM calls can take time
                            )
                            if ui_response.status_code == 200:
                                parsed_layout = ui_response.json()

                                # parsed_layout may be None if agent failed
                                if not parsed_layout:
                                    logger.warning(f"UI builder agent returned empty/None layout for project {payload.project_id} — skipping canvas update")
                                else:
                                    # Support new pages[] schema and old nodes[] schema
                                    # New: { "pages": [...], "edges": [...] }
                                    # Old: { "layout_json": { "nodes": [...], "edges": [...] } }
                                    if "pages" in parsed_layout:
                                        layout_data = parsed_layout  # new schema — store as-is
                                    elif "layout_json" in parsed_layout:
                                        layout_data = parsed_layout["layout_json"]
                                        # Normalise old nodes[] entries
                                        for idx, nd in enumerate(layout_data.get("nodes", [])):
                                            if "position" not in nd:
                                                nd["position"] = {"x": 100 + (idx * 280), "y": 150}
                                            nd.setdefault("type", "custom")
                                            nd.setdefault("data", {"label": nd.get("id", f"Screen {idx + 1}"), "description": "Mockup screen"})
                                    else:
                                        layout_data = parsed_layout  # fallback — store whatever came back

                                    # Only persist when we have actual page content
                                    has_pages = len(layout_data.get("pages", [])) > 0 or len(layout_data.get("nodes", [])) > 0
                                    if not has_pages:
                                        logger.warning(f"UI builder returned layout with no pages for project {payload.project_id} — skipping canvas update")
                                    else:
                                        # Persist to MongoDB canvases collection
                                        existing_canvas = await db.canvases.find_one({"project_id": payload.project_id})
                                        if existing_canvas:
                                            await db.canvases.update_one(
                                                {"project_id": payload.project_id},
                                                {"$set": {"layout_json": layout_data, "updated_at": datetime.utcnow()}}
                                            )
                                            logger.info(f"Canvas updated for project {payload.project_id} with {len(layout_data.get('pages', []))} pages")
                                        else:
                                            await db.canvases.insert_one({
                                                "tenant_id": tenant_id,
                                                "project_id": payload.project_id,
                                                "layout_json": layout_data,
                                                "created_at": datetime.utcnow(),
                                                "updated_at": datetime.utcnow()
                                            })
                                            logger.info(f"Canvas created for project {payload.project_id}")
                            else:
                                logger.error(f"UI builder agent returned status {ui_response.status_code} for project {payload.project_id}")
                    except Exception as ui_err:
                        logger.error(f"Failed to generate UI mockup from PRD: {str(ui_err)}")
            else:
                # In CHAT mode: we do NOT overwrite or edit the PRD document body in MongoDB, but we DO save conversation messages to chat_history!
                await _set_job(job_id, {"explainability": "Committing conversation logs to database..."})
                await DocsManagementService.update_document(
                    document_id=doc_id,
                    payload={}, 
                    agent_chat_msg=f"User: {payload.messages[-1].content if payload.messages else ''} | Agent: {ai_text_reply}"
                )

            # Commit the fully unique response and summary data from the agent to active storage
            await _set_job(job_id, {
                "status": "completed",
                "explainability": "Generation task completed successfully.",
                "response": ai_text_reply,
                "change_summary": dynamic_summary,
                "is_complete": is_complete_val
            })

        except Exception as async_err:
            await _set_job(job_id, {"status": "failed", "explainability": f"Error: {str(async_err)}"})

@router.post("/interactive-chat", response_model=InteractivePRDResponse)
async def process_prd_chat_step(payload: InteractivePRDRequest, background_tasks: BackgroundTasks, tenant_id: str = Depends(resolve_tenant)):
    doc_id = payload.document_id
    target_title = f"Interactive PRD: {payload.product_name}"

    if not doc_id:
        db = get_db()
        existing_doc = await db.documents.find_one({"tenant_id": tenant_id, "workspace_id": payload.workspace_id, "title": target_title})
        if existing_doc:
            doc_id = str(existing_doc["_id"])
        else:
            created_doc = await DocsManagementService.create_document(tenant_id=tenant_id, workspace_id=payload.workspace_id, payload={"title": target_title, "body": "Initializing configuration loop..."})
            doc_id = str(created_doc["_id"])

    job_id = str(uuid.uuid4())
    await _set_job(job_id, {"status": "processing", "explainability": "Spawning async background thread execution tracks..."})

    if settings.background_jobs_enabled:
        background_tasks.add_task(execute_async_langgraph_pipeline, job_id, tenant_id, payload, doc_id)
        return {"job_id": job_id, "document_id": doc_id, "status": "processing"}

    # Serverless mode (DV_BACKGROUND_JOBS_ENABLED=false): run inline and persist
    # the result so the polling endpoint can serve it from MongoDB.
    await execute_async_langgraph_pipeline(job_id, tenant_id, payload, doc_id)
    job_info = await _get_job(job_id) or {}
    return {"job_id": job_id, "document_id": doc_id, "status": job_info.get("status", "processing")}

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status_tracking_data(job_id: str):
    job_info = await _get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job footprint missing.")
    
    return {
        "job_id": job_id,
        "status": job_info.get("status", "processing"),
        "explainability": job_info.get("explainability", "Processing computational inference runs..."), #  Streamed from agent
        "response": job_info.get("response"),
        "change_summary": job_info.get("change_summary"), #  Generated dynamically by agent
        "is_complete": job_info.get("is_complete", False)
    }