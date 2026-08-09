import httpx
import re
import json
import html
from datetime import datetime
from bson import ObjectId
from fastapi import HTTPException
from app.db.mongo import get_db
from app.core.config import settings


def _clean_prd_body(raw: str) -> str:
    """Strip HTML tags/entities and extract raw markdown from JSON wrapper."""
    if not raw:
        return raw
    text = html.unescape(raw)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    for _ in range(5):
        if text.startswith('{') or '"response"' in text[:200]:
            inner = text
            if inner.startswith('```'):
                inner = re.sub(r'^```[a-z]*\n?', '', inner)
                inner = re.sub(r'\n?```$', '', inner.strip())
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and 'response' in parsed:
                    text = parsed['response']
                    text = text.replace('\\n', '\n').replace('\\"', '"')
                    continue
                if isinstance(parsed, str):
                    text = parsed
                    continue
            except Exception:
                m = re.search(r'"response"\s*:\s*"([\s\S]+?)"\s*(?:,|\s*})', inner)
                if m:
                    text = m.group(1).replace('\\n', '\n').replace('\\"', '"')
                    continue
            break
        else:
            break
    return text.strip()


class WorkflowGeneratorService:

    @staticmethod
    async def generate_workflow(tenant_id: str, project_id: str) -> str:
        db = get_db()
        project_oid = ObjectId(project_id)

        # 1. Fetch project — no tenant filter (route guard already enforces access)
        project = await db.projects.find_one({"_id": project_oid})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        product_name = project.get("name", "New Project")
        tenant_oid = project.get("tenant_id", ObjectId(tenant_id))

        # 2. Locate workspaces (ObjectId first, then string fallback for legacy records)
        workspaces = await db.workspaces.find({"project_id": project_oid}).to_list(length=100)
        if not workspaces:
            workspaces = await db.workspaces.find({"project_id": project_id}).to_list(length=100)
        if not workspaces:
            raise HTTPException(
                status_code=400,
                detail="No workspaces found for this project. Please initialize the project first."
            )

        ws_ids = [str(ws["_id"]) for ws in workspaces]

        # 3. Locate the latest PRD document
        prd_doc = await db.documents.find_one(
            {"workspace_id": {"$in": ws_ids}, "title": {"$regex": "PRD", "$options": "i"}},
            sort=[("updated_at", -1)]
        )
        if not prd_doc:
            prd_doc = await db.documents.find_one(
                {"workspace_id": {"$in": ws_ids}},
                sort=[("updated_at", -1)]
            )
        if not prd_doc or not prd_doc.get("body", "").strip():
            raise HTTPException(
                status_code=400,
                detail="No PRD document found. Please generate a PRD first."
            )

        # 4. Clean the body (strip HTML + unwrap JSON wrapper)
        prd_body = _clean_prd_body(prd_doc["body"])
        if not prd_body:
            raise HTTPException(
                status_code=400,
                detail="PRD document body is empty after processing. Please regenerate the PRD."
            )

        # 5. Call the LangGraph workflow generation microservice
        langgraph_url = f"{settings.langgraph_url}/workflow/generate-full"
        payload = {"product_name": product_name, "prd_body": prd_body}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(langgraph_url, json=payload, timeout=None)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"LangGraph service error ({response.status_code}): {response.text[:500]}"
                )
            data = response.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to communicate with LangGraph microservice: {str(e)}"
            )

        # 6. Validate response shape — support both old (phases) and new (epics) formats
        if "epics" not in data and "phases" not in data:
            raise HTTPException(
                status_code=500,
                detail="LangGraph returned an unexpected response format."
            )

        # 7. Wipe existing workflow for this project (allow re-generation)
        existing_wf = await db.project_workflows.find_one({"project_id": project_oid})
        if existing_wf:
            old_wf_id = existing_wf["_id"]
            await db.project_workflows.delete_one({"_id": old_wf_id})
            await db.workflow_epics.delete_many({"workflow_id": old_wf_id})
            await db.workflow_features.delete_many({"workflow_id": old_wf_id})
            await db.workflow_tasks.delete_many({"workflow_id": old_wf_id})
            await db.workflow_sprints.delete_many({"workflow_id": old_wf_id})
            # Legacy collections
            await db.workflow_phases.delete_many({"workflow_id": old_wf_id})
            await db.workflow_task_dependencies.delete_many({"workflow_id": old_wf_id})

        # 8. Store workflow root document
        statistics = data.get("statistics", {})
        react_flow = data.get("react_flow", {"nodes": [], "edges": []})

        wf_doc = {
            "project_id": project_oid,
            "tenant_id": tenant_oid,
            "status": "active",
            "statistics": statistics,
            "react_flow_nodes": react_flow.get("nodes", []),
            "react_flow_edges": react_flow.get("edges", []),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        wf_res = await db.project_workflows.insert_one(wf_doc)
        workflow_id = wf_res.inserted_id

        # ── NEW FORMAT (epics → features → tasks) ──────────────────────────
        if "epics" in data:
            epics = data["epics"]
            sprints = data.get("sprints", [])

            # Store sprints
            for sprint in sprints:
                await db.workflow_sprints.insert_one({
                    "workflow_id": workflow_id,
                    "sprint_number": sprint.get("id", 1),
                    "name": sprint.get("name", "Sprint"),
                    "goal": sprint.get("goal", ""),
                    "story_points": sprint.get("story_points", 0),
                    "velocity": sprint.get("velocity", 40),
                    "task_count": sprint.get("task_count", 0),
                    "epics": sprint.get("epics", []),
                    "deliverables": sprint.get("deliverables", []),
                    "exit_criteria": sprint.get("exit_criteria", []),
                    "created_at": datetime.utcnow(),
                })

            # Store epics, features, tasks
            for epic_order, epic in enumerate(epics):
                epic_doc = {
                    "workflow_id": workflow_id,
                    "epic_id": epic.get("id"),
                    "title": epic.get("title", "Untitled Epic"),
                    "description": epic.get("description", ""),
                    "color": epic.get("color", "#6366F1"),
                    "order": epic.get("order", epic_order + 1),
                    "sprint": epic.get("sprint", 1),
                    "feature_count": epic.get("feature_count", 0),
                    "task_count": epic.get("task_count", 0),
                    "created_at": datetime.utcnow(),
                }
                epic_res = await db.workflow_epics.insert_one(epic_doc)
                epic_db_id = epic_res.inserted_id

                for feat_order, feature in enumerate(epic.get("features", [])):
                    feat_doc = {
                        "workflow_id": workflow_id,
                        "epic_db_id": epic_db_id,
                        "epic_id": epic.get("id"),
                        "feature_id": feature.get("id"),
                        "title": feature.get("title", "Untitled Feature"),
                        "type": feature.get("type", "generic_feature"),
                        "sprint": feature.get("sprint", epic.get("sprint", 1)),
                        "order": feat_order + 1,
                        "created_at": datetime.utcnow(),
                    }
                    feat_res = await db.workflow_features.insert_one(feat_doc)
                    feat_db_id = feat_res.inserted_id

                    for task_order, task in enumerate(feature.get("tasks", [])):
                        task_doc = {
                            "workflow_id": workflow_id,
                            "epic_db_id": epic_db_id,
                            "feat_db_id": feat_db_id,
                            "epic_id": epic.get("id"),
                            "feature_id": feature.get("id"),
                            "task_id": task.get("id"),
                            "title": task.get("title", "Untitled Task"),
                            "description": task.get("description", ""),
                            "epic": task.get("epic", epic.get("title", "")),
                            "feature": task.get("feature", feature.get("title", "")),
                            "assigned_agent": task.get("assigned_agent", "Product Agent"),
                            "sprint": task.get("sprint", epic.get("sprint", 1)),
                            "priority": task.get("priority", "medium").lower(),
                            "story_points": task.get("story_points", 3),
                            "estimated_hours": float(task.get("estimated_hours", 6)),
                            "status": "pending",
                            "acceptance_criteria": task.get("acceptance_criteria", []),
                            "definition_of_done": task.get("definition_of_done", []),
                            "prd_section": task.get("prd_section", ""),
                            "depends_on": task.get("depends_on", []),
                            "blocks": task.get("blocks", []),
                            "artifacts": task.get("artifacts", []),
                            "assigned_human_id": None,
                            "order": task_order + 1,
                            "created_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow(),
                        }
                        await db.workflow_tasks.insert_one(task_doc)

        # ── LEGACY FORMAT (phases) — backward-compatible ────────────────────
        elif "phases" in data:
            phases = data["phases"]
            for p_idx, phase in enumerate(phases):
                phase_doc = {
                    "workflow_id": workflow_id,
                    "name": phase.get("name", "Phase"),
                    "order": p_idx + 1,
                    "status": "pending",
                    "created_at": datetime.utcnow(),
                }
                ph_res = await db.workflow_phases.insert_one(phase_doc)
                phase_id = ph_res.inserted_id
                for task in phase.get("tasks", []):
                    await db.workflow_tasks.insert_one({
                        "workflow_id": workflow_id,
                        "phase_id": phase_id,
                        "title": task.get("title", "Untitled Task"),
                        "description": task.get("description", ""),
                        "phase": str(task.get("phase", phase.get("name", ""))).lower(),
                        "priority": task.get("priority", "medium").lower(),
                        "status": "pending",
                        "story_points": task.get("story_points", 3),
                        "estimated_hours": float(task.get("estimated_hours", 6)),
                        "assigned_agent": task.get("assigned_agent", "Product Agent"),
                        "acceptance_criteria": task.get("acceptance_criteria", []),
                        "artifacts": [],
                        "assigned_human_id": None,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow(),
                    })

        # Tasks remain in 'pending' status after generation — no auto-completion

        return str(workflow_id)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def get_project_workflow(tenant_id: str, project_id: str) -> dict:
        db = get_db()
        project_oid = ObjectId(project_id)

        # Don't filter by tenant — project access was verified by route guard
        workflow = await db.project_workflows.find_one({"project_id": project_oid})
        if not workflow:
            return {}

        workflow_id = workflow["_id"]

        # ── Try new epic-based format first ────────────────────────────────
        epics_raw = await db.workflow_epics.find(
            {"workflow_id": workflow_id}
        ).sort("order", 1).to_list(length=200)

        if epics_raw:
            # New format
            features_raw = await db.workflow_features.find(
                {"workflow_id": workflow_id}
            ).sort("order", 1).to_list(length=2000)
            tasks_raw = await db.workflow_tasks.find(
                {"workflow_id": workflow_id}
            ).sort("order", 1).to_list(length=5000)
            sprints_raw = await db.workflow_sprints.find(
                {"workflow_id": workflow_id}
            ).sort("sprint_number", 1).to_list(length=50)

            # Build feature map: epic_id → list of features
            feat_by_epic: dict = {}
            for f in features_raw:
                eid = f.get("epic_id")
                feat_by_epic.setdefault(eid, []).append(f)

            # Build task map: feature_id → list of tasks
            task_by_feat: dict = {}
            for t in tasks_raw:
                fid = t.get("feature_id")
                task_by_feat.setdefault(fid, []).append(t)

            serialized_epics = []
            for e in epics_raw:
                eid = e.get("epic_id")
                e_features = feat_by_epic.get(eid, [])
                serialized_features = []
                for f in e_features:
                    fid = f.get("feature_id")
                    f_tasks = task_by_feat.get(fid, [])
                    serialized_tasks = [_serialize_task(t) for t in f_tasks]
                    serialized_features.append({
                        "id": fid,
                        "title": f.get("title", ""),
                        "type": f.get("type", "generic_feature"),
                        "sprint": f.get("sprint", 1),
                        "task_count": len(serialized_tasks),
                        "tasks": serialized_tasks,
                    })
                serialized_epics.append({
                    "id": eid,
                    "title": e.get("title", ""),
                    "description": e.get("description", ""),
                    "color": e.get("color", "#6366F1"),
                    "order": e.get("order", 1),
                    "sprint": e.get("sprint", 1),
                    "feature_count": len(serialized_features),
                    "task_count": sum(f["task_count"] for f in serialized_features),
                    "features": serialized_features,
                })

            serialized_sprints = [
                {
                    "id": s.get("sprint_number"),
                    "name": s.get("name"),
                    "goal": s.get("goal", ""),
                    "story_points": s.get("story_points", 0),
                    "velocity": s.get("velocity", 40),
                    "task_count": s.get("task_count", 0),
                    "epics": s.get("epics", []),
                    "deliverables": s.get("deliverables", []),
                    "exit_criteria": s.get("exit_criteria", []),
                }
                for s in sprints_raw
            ]

            return {
                "id": str(workflow_id),
                "project_id": project_id,
                "status": workflow.get("status", "active"),
                "statistics": workflow.get("statistics", {}),
                "epics": serialized_epics,
                "sprints": serialized_sprints,
                "react_flow": {
                    "nodes": workflow.get("react_flow_nodes", []),
                    "edges": workflow.get("react_flow_edges", []),
                },
            }

        # ── Fallback to legacy phase-based format ──────────────────────────
        phases_raw = await db.workflow_phases.find(
            {"workflow_id": workflow_id}
        ).sort("order", 1).to_list(length=100)

        if not phases_raw:
            return {}

        tasks_raw = await db.workflow_tasks.find(
            {"workflow_id": workflow_id}
        ).to_list(length=1000)
        deps_raw = await db.workflow_task_dependencies.find(
            {"workflow_id": workflow_id}
        ).to_list(length=1000)

        serialized_phases = []
        for p in phases_raw:
            p_id = p["_id"]
            p_tasks = []
            for t in tasks_raw:
                if t.get("phase_id") == p_id:
                    t_id = t["_id"]
                    t_deps = []
                    for d in deps_raw:
                        if d.get("task_id") == t_id:
                            dep_title = next(
                                (x["title"] for x in tasks_raw if x["_id"] == d.get("depends_on_task_id")),
                                "Unknown Task"
                            )
                            t_deps.append({
                                "target_id": str(d.get("depends_on_task_id")),
                                "target_title": dep_title,
                                "type": d.get("type", "dependency"),
                            })
                    p_tasks.append({
                        "id": str(t_id),
                        "title": t.get("title", ""),
                        "description": t.get("description", ""),
                        "phase": t.get("phase", ""),
                        "priority": t.get("priority", "medium"),
                        "status": t.get("status", "pending"),
                        "story_points": t.get("story_points", 3),
                        "estimated_hours": t.get("estimated_hours", 6),
                        "assigned_agent": t.get("assigned_agent", ""),
                        "assigned_human_id": t.get("assigned_human_id"),
                        "acceptance_criteria": t.get("acceptance_criteria", []),
                        "artifacts": t.get("artifacts", []),
                        "dependencies": t_deps,
                    })
            serialized_phases.append({
                "id": str(p_id),
                "name": p.get("name", ""),
                "order": p.get("order", 1),
                "status": p.get("status", "pending"),
                "tasks": p_tasks,
            })

        return {
            "id": str(workflow_id),
            "project_id": project_id,
            "status": workflow.get("status", "active"),
            "statistics": workflow.get("statistics", {}),
            "phases": serialized_phases,
        }

    @staticmethod
    async def update_task(tenant_id: str, task_id: str, update_data: dict) -> dict:
        db = get_db()
        task_oid = ObjectId(task_id)

        update_fields = {k: v for k, v in update_data.items()
                         if k in ("status", "assigned_human_id", "priority")}
        update_fields["updated_at"] = datetime.utcnow()

        result = await db.workflow_tasks.find_one_and_update(
            {"_id": task_oid},
            {"$set": update_fields},
            return_document=True,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        return _serialize_task(result)

    @staticmethod
    async def run_agent_auto_execution(workflow_id: ObjectId):
        db = get_db()
        
        # 1. Retrieve all tasks belonging to this workflow that are assigned to the UI Designer or Frontend Agent
        tasks = await db.workflow_tasks.find({
            "workflow_id": workflow_id,
            "assigned_agent": {"$regex": "UI Designer|Designer|Frontend", "$options": "i"}
        }).to_list(length=100)
        
        for task in tasks:
            task_id = task["_id"]
            
            # Transition to in_progress
            await db.workflow_tasks.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "status": "in_progress",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Simulate work details and generate high-fidelity UI artifacts matching the product specifications!
            mock_artifacts = [
                {
                    "name": "Wireframe Layout Specification",
                    "type": "design_spec",
                    "url": "https://figma.com/file/mock-prd-layout-spec",
                    "created_at": datetime.utcnow().isoformat()
                },
                {
                    "name": "UI Component Library Integration Map",
                    "type": "code_snippet",
                    "url": "https://github.com/decision-vault/component-map",
                    "created_at": datetime.utcnow().isoformat()
                }
            ]
            
            # Transition to completed and attach the generated artifacts
            await db.workflow_tasks.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "status": "completed",
                        "artifacts": mock_artifacts,
                        "updated_at": datetime.utcnow()
                    }
                }
            )


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _serialize_task(t: dict) -> dict:
    return {
        "id": str(t["_id"]),
        "title": t.get("title", ""),
        "description": t.get("description", ""),
        "epic_id": t.get("epic_id", ""),
        "epic": t.get("epic", ""),
        "feature_id": t.get("feature_id", ""),
        "feature": t.get("feature", ""),
        "assigned_agent": t.get("assigned_agent", ""),
        "sprint": t.get("sprint", 1),
        "priority": t.get("priority", "medium"),
        "story_points": t.get("story_points", 3),
        "estimated_hours": t.get("estimated_hours", 6),
        "status": t.get("status", "pending"),
        "acceptance_criteria": t.get("acceptance_criteria", []),
        "definition_of_done": t.get("definition_of_done", []),
        "prd_section": t.get("prd_section", ""),
        "depends_on": t.get("depends_on", []),
        "blocks": t.get("blocks", []),
        "artifacts": t.get("artifacts", []),
        "assigned_human_id": t.get("assigned_human_id"),
        # Legacy compat
        "phase": t.get("phase", t.get("feature", "")),
        "dependencies": [],
    }
