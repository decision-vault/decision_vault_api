import traceback
import logging
import uuid
import json
import re
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx
from app.schemas.prd_generator_schemas import InteractivePRDRequest, InteractivePRDResponse, JobStatusResponse
from app.services.docs_management_service import DocsManagementService
from app.middleware.tenant import resolve_tenant
from app.core.config import settings
from app.db.mongo import get_db
from app.services.usage_service import estimate_tokens, record_usage

logger = logging.getLogger("decisionvault.prd_generator")
router = APIRouter(prefix="/api/prd-generator", tags=["AI PRD Automation Engine"])

from bson import ObjectId
from datetime import datetime

ACTIVE_PROCESSING_JOBS = {}
JOB_EVENT_QUEUES: dict[str, asyncio.Queue] = {}


def _is_document_snapshot(text: str) -> bool:
    """Heuristic: a full generated document (PRD / strategy analysis) vs. a short chat reply."""
    stripped = (text or "").strip()
    if len(stripped) <= 400:
        return False
    headings = len(re.findall(r"^#{1,3} ", stripped, flags=re.MULTILINE))
    return headings >= 3


def _summarize_execution_steps(events: list) -> list:
    """Rebuild the final per-step statuses from the persisted structured event log."""
    steps: dict[str, dict] = {}
    for ev in events or []:
        event_type = ev.get("type")
        step_id = ev.get("step_id")
        if event_type == "plan":
            for s in ev.get("steps") or []:
                steps[s.get("id")] = {
                    "id": s.get("id"), "title": s.get("title", s.get("id")),
                    "description": s.get("description", ""), "status": "pending",
                }
        elif event_type == "step.started" and step_id:
            st = steps.setdefault(step_id, {"id": step_id, "title": ev.get("title", step_id), "description": ev.get("description", ""), "status": "pending"})
            st["status"] = "active"
        elif event_type == "step.progress" and step_id:
            st = steps.setdefault(step_id, {"id": step_id, "title": ev.get("title", step_id), "status": "active"})
            if ev.get("status"):
                st["progress"] = str(ev["status"])
        elif event_type == "step.completed" and step_id:
            st = steps.setdefault(step_id, {"id": step_id, "title": ev.get("title", step_id), "status": "done"})
            st["status"] = "done"
            if ev.get("result"):
                st["result"] = str(ev["result"])[:500]
        elif event_type == "step.failed" and step_id:
            st = steps.setdefault(step_id, {"id": step_id, "title": ev.get("title", step_id), "status": "failed"})
            st["status"] = "failed"
            st["message"] = str(ev.get("message") or "")
    return list(steps.values())

# Structured execution events the agent emits (SSE frame `t` value → event type).
STRUCTURED_EVENTS = {
    "execution.started",
    "plan",
    "step.started",
    "step.progress",
    "step.completed",
    "step.failed",
    "ai_response",
    "execution.completed",
    "execution.failed",
}

# Map step/plan events onto the legacy `explainability` string so the old
# polling UI and the process tab keep working alongside the new activity UI.
def _explainability_for_event(event_type: str, data: dict) -> str | None:
    if event_type == "execution.started":
        return "Agent started."
    if event_type == "plan":
        steps = data.get("steps") or []
        return f"Planning {len(steps)} analysis steps..."
    if event_type == "step.started":
        return f"{data.get('title', 'Processing')}..."
    if event_type == "step.progress":
        return str(data.get("status") or data.get("detail") or "Working...")
    if event_type == "step.completed":
        return f"{data.get('title', 'Step')} ✓"
    if event_type in ("ai_response", "execution.completed"):
        return "Generation complete."
    if event_type == "execution.failed":
        return f"Agent failed: {str(data.get('message', ''))[:500]}"
    return None


async def _record_event(job_id: str, event_type: str, data: dict) -> None:
    """Persist a structured execution event with a monotonic sequence number and
    fan it out to any live /jobs/{id}/stream subscribers."""
    ACTIVE_PROCESSING_JOBS.setdefault(job_id, {})
    ACTIVE_PROCESSING_JOBS[job_id].setdefault("_seq", 0)
    ACTIVE_PROCESSING_JOBS[job_id]["_seq"] += 1
    seq = ACTIVE_PROCESSING_JOBS[job_id]["_seq"]
    event = {"seq": seq, "type": event_type, **data,
             "timestamp": datetime.utcnow().isoformat() + "Z"}
    ACTIVE_PROCESSING_JOBS[job_id].setdefault("events", []).append(event)

    try:
        db = get_db()
        await db.prd_generation_jobs.update_one(
            {"job_id": job_id},
            {"$push": {"events": event}},
            upsert=True,
        )
    except Exception as persist_err:
        logger.warning(f"Failed to persist event {event_type} for {job_id}: {persist_err}")

    q = JOB_EVENT_QUEUES.get(job_id)
    if q is not None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    explainability = _explainability_for_event(event_type, data)
    if explainability is not None:
        await _set_job(job_id, {"explainability": explainability})

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


async def _try_streaming_agent(job_id: str, micro_payload: dict):
    """Start the interactive graph agent as a streaming subprocess and consume
    its SSE frames, keeping the job's explainability field live.

    Returns (result_payload | None, fallback_allowed: bool). fallback_allowed is
    False when the agent already started and failed mid-stream, so the caller
    must not fall back to the synchronous call (that would double-run it).
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            start_resp = await client.post(
                f"{settings.langgraph_url}/api/agent/execute",
                json=micro_payload,
            )
            start_resp.raise_for_status()
            execution_id = (start_resp.json() or {}).get("execution_id")
        if not execution_id:
            return None, True
    except Exception as e:
        logger.warning(f"Agent streaming execution unavailable: {e}")
        return None, True

    await _set_job(job_id, {"execution_id": execution_id, "explainability": "Agent spawned — streaming generation status..."})

    result_payload = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1200.0)) as client:
        try:
            async with client.stream("GET", f"{settings.langgraph_url}/api/agent/execute/{execution_id}/stream") as resp:
                resp.raise_for_status()
                event_type = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if not data:
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        kind = event_type
                        if kind == "status":
                            label = str(event.get("label") or "")
                            detail = str(event.get("detail") or "")
                            text = f"{label} {detail}".strip()[:500] or "Agent is generating..."
                            await _set_job(job_id, {"explainability": text})
                        elif kind in STRUCTURED_EVENTS:
                            structured_payload = {"execution_id": execution_id}
                            structured_payload.update({k: v for k, v in event.items() if k != "execution_id"})
                            await _record_event(job_id, kind, structured_payload)
                        elif kind == "error":
                            await _record_event(job_id, "execution.failed", {"execution_id": execution_id, "message": str(event.get("message", ""))})
                            await _set_job(job_id, {"status": "failed", "explainability": f"Agent error: {str(event.get('message', ''))[:500]}"})
                            return None, False
                        elif kind in ("result", "done"):
                            r = event.get("result") or {}
                            if r.get("response") is not None:
                                result_payload = {
                                    "response": r.get("response", ""),
                                    "change_summary": r.get("change_summary", "Refined functional specification document blocks."),
                                    "is_complete": r.get("is_complete", False),
                                }
                            if kind == "done":
                                break
        except Exception as e:
            logger.error(f"Agent streaming failed: {e}")
            return None, False

    if not result_payload:
        await _set_job(job_id, {"status": "failed", "explainability": "Agent returned no result."})
        return None, False
    return result_payload, True


async def _finalize_job(job_id: str, tenant_id: str, payload, doc_id: str, citations: list, result_payload: dict) -> None:
    """Shared post-processing + completion for both the streaming and legacy paths."""
    ai_text_reply = result_payload.get("response", "")
    dynamic_summary = result_payload.get("change_summary", "Refined functional specification document blocks.")
    is_complete_val = result_payload.get("is_complete", False)

    await record_usage(
        tenant_id,
        project_id=payload.project_id,
        feature="prd_generator",
        prompt_tokens=estimate_tokens(json.dumps(payload.model_dump())),
        completion_tokens=estimate_tokens(ai_text_reply),
    )

    if payload.mode == "plan":
        await _set_job(job_id, {"explainability": "Committing section mutations to core database models..."})
        db = get_db()

        await DocsManagementService.update_document(
            document_id=doc_id,
            payload={"body": ai_text_reply},
            agent_chat_msg=f"User: {payload.messages[-1].content if payload.messages else ''} | Agent: {ai_text_reply}",
            is_doc_snapshot=True,
            snapshot_body=ai_text_reply
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

                        await record_usage(
                            tenant_id,
                            project_id=payload.project_id,
                            feature="ui_builder",
                            prompt_tokens=estimate_tokens(ui_payload.get("prd_body") or ""),
                            completion_tokens=estimate_tokens(json.dumps(parsed_layout) if parsed_layout else ""),
                        )

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
        is_doc_snapshot = _is_document_snapshot(ai_text_reply)
        await DocsManagementService.update_document(
            document_id=doc_id,
            payload={},
            agent_chat_msg=f"User: {payload.messages[-1].content if payload.messages else ''} | Agent: {ai_text_reply}",
            citations=citations or None,
            is_doc_snapshot=is_doc_snapshot,
            snapshot_body=ai_text_reply if is_doc_snapshot else None
        )

    # Commit the fully unique response and summary data from the agent to active storage
    await _set_job(job_id, {
        "status": "completed",
        "explainability": "Generation task completed successfully.",
        "response": ai_text_reply,
        "change_summary": dynamic_summary,
        "is_complete": is_complete_val,
        "citations": citations or []
    })

    # Persist the last execution summary on the document so the UI can restore
    # the activity panel (execution steps + response) after a page refresh.
    try:
        job_record = await get_db().prd_generation_jobs.find_one({"job_id": job_id})
        events = (job_record or {}).get("events") or []
        await get_db().documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "last_execution": {
                    "job_id": job_id,
                    "execution_id": (job_record or {}).get("execution_id") or ACTIVE_PROCESSING_JOBS.get(job_id, {}).get("execution_id"),
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "steps": _summarize_execution_steps(events),
                    "response": ai_text_reply,
                },
                "updated_at": datetime.utcnow(),
            }}
        )
    except Exception as exec_persist_err:
        logger.warning(f"Failed to persist last_execution for doc {doc_id}: {exec_persist_err}")

async def execute_async_langgraph_pipeline(job_id: str, tenant_id: str, payload: InteractivePRDRequest, doc_id: str):
    langgraph_url = f"{settings.langgraph_url}/workflow/prd/interactive-step"
    db = get_db()
    
    messages_payload = []
    citations = []

    # 1. Grounded chat mode: retrieve from the project knowledge base and inject
    #    as QA context. Plan mode keeps the full document body so the agent can
    #    rewrite sections in place.
    if payload.mode == "chat" and payload.project_id:
        try:
            from app.services import knowledge_service
            await knowledge_service.ensure_indexed(tenant_id, payload.project_id)
            last_user = next(
                (m.content for m in reversed(payload.messages) if m.role == "user"),
                payload.messages[-1].content if payload.messages else "",
            )
            context_msgs, citations = await knowledge_service.build_grounded_context(
                tenant_id, payload.project_id, last_user, top_k=4
            )
            messages_payload.extend(context_msgs)
        except Exception as kb_err:
            logger.error(f"Failed to build grounded knowledge context: {str(kb_err)}")
            citations = []

    # 1b. Fallback for chat without a project, and plan mode: supply the current
    #     document specifications as QA context.
    if not messages_payload:
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

    # ── Streaming agent execution (live status frames) with legacy fallback ──
    result_payload, fallback_allowed = await _try_streaming_agent(job_id, micro_payload)

    if result_payload is None and fallback_allowed:
        await _set_job(job_id, {"explainability": "Streaming unavailable — falling back to synchronous agent call..."})
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(langgraph_url, json=micro_payload, timeout=None)

                if response.status_code != 200:
                    await _set_job(job_id, {"status": "failed", "explainability": "Crashed downstream."})
                    return

                data = response.json()

            result_payload = {
                "response": data.get("response", ""),
                "change_summary": data.get("change_summary", "Refined functional specification document blocks."),
                "is_complete": data.get("is_complete", False),
            }
        except Exception as async_err:
            await _set_job(job_id, {"status": "failed", "explainability": f"Error: {str(async_err)}"})
            return

    if result_payload is not None:
        await _finalize_job(job_id, tenant_id, payload, doc_id, citations, result_payload)

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
        "is_complete": job_info.get("is_complete", False),
        "citations": job_info.get("citations", [])
    }
def _format_event(event: dict) -> str:
    data = {k: v for k, v in event.items() if k not in ("seq",)}
    return f"id: {event.get('seq', 0)}\nevent: {event.get('type', 'status')}\ndata: {json.dumps(data)}\n\n"


@router.get("/jobs/{job_id}/stream")
async def get_job_stream(job_id: str):
    """SSE stream that tails a background job in REAL TIME.

    Structured execution events are emitted as they are produced by the agent
    (event types: execution.started, plan, step.started, step.progress,
    step.completed, step.failed, ai_response, execution.completed,
    execution.failed), each carrying an `id` sequence for dedup. Events that
    occurred before this stream connected are replayed from the job store, so
    a late subscriber still sees the full timeline.

    Legacy frames (status / result / error / done) are still emitted for
    backward compatibility with the polling UI.
    """
    async def event_generator():
        q = asyncio.Queue(maxsize=200)
        JOB_EVENT_QUEUES[job_id] = q
        try:
            job_info = await _get_job(job_id)
            if job_info is None:
                yield 'event: error\ndata: {"message":"Job footprint missing."}\n\n'
                yield 'event: done\ndata: {}\n\n'
                return

            seen_seqs = set()
            for event in job_info.get("events") or []:
                seq = event.get("seq", 0)
                if seq in seen_seqs:
                    continue
                seen_seqs.add(seq)
                yield _format_event(event)
                if event.get("type") in ("execution.completed", "execution.failed"):
                    job_info = await _get_job(job_id) or {}
                    if job_info.get("status") == "completed":
                        yield f"event: result\ndata: {json.dumps({'response': job_info.get('response'), 'change_summary': job_info.get('change_summary'), 'is_complete': job_info.get('is_complete', False), 'citations': job_info.get('citations') or []})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return
                    if job_info.get("status") == "failed":
                        yield f"event: error\ndata: {json.dumps({'message': job_info.get('explainability', 'Agent failed.')})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
                    fresh = await _get_job(job_id) or {}
                    if fresh.get("status") == "completed":
                        yield f"event: result\ndata: {json.dumps({'response': fresh.get('response'), 'change_summary': fresh.get('change_summary'), 'is_complete': fresh.get('is_complete', False), 'citations': fresh.get('citations') or []})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return
                    if fresh.get("status") == "failed":
                        yield f"event: error\ndata: {json.dumps({'message': fresh.get('explainability', 'Agent failed.')})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return
                    continue
                seq = event.get("seq", 0)
                if seq in seen_seqs:
                    continue
                seen_seqs.add(seq)
                yield _format_event(event)
                if event.get("type") in ("execution.completed", "execution.failed"):
                    fresh = await _get_job(job_id) or {}
                    if fresh.get("status") == "completed":
                        yield f"event: result\ndata: {json.dumps({'response': fresh.get('response'), 'change_summary': fresh.get('change_summary'), 'is_complete': fresh.get('is_complete', False), 'citations': fresh.get('citations') or []})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return
                    if fresh.get("status") == "failed":
                        yield f"event: error\ndata: {json.dumps({'message': fresh.get('explainability', 'Agent failed.')})}\n\n"
                        yield 'event: done\ndata: {}\n\n'
                        return
        finally:
            JOB_EVENT_QUEUES.pop(job_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents/{document_id}/chat")
async def get_document_chat_history(
    document_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(8, ge=1, le=50),
):
    """Paginated chat history for a document (newest-first pages).

    Page 1 returns the newest `page_size` entries; increasing `page` returns
    progressively older entries. `has_more` indicates older entries exist.
    Each entry is returned as-is (including `is_doc_snapshot` and
    `saved_snapshot_body` so individual generated documents can be reopened).
    """
    if not ObjectId.is_valid(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    db = get_db()
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    entries = doc.get("chat_history") or []
    total = len(entries)
    end = total - (page - 1) * page_size
    start = max(0, end - page_size)
    page_entries = entries[start:end] if start < end else []

    return {
        "document_id": document_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": start > 0,
        "entries": page_entries,
    }

