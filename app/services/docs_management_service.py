import logging
from bson import ObjectId
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.db.mongo import get_db
logger = logging.getLogger("decisionvault.docs_management")
class DocsManagementService:
    @staticmethod
    async def initialize_project_workspace(tenant_id: str, project_id: str, project_name: str) -> Dict[str, Any]:
        """
        Automated Hooks: Instantly provisions a default workspace and baseline 
        PRD anchor document upon success of any parent project initialization loops.
        """
        db = get_db()
        
        # 1. Spawn matching default workspace partition
        new_ws = {
            "name": f"{project_name} Workspace",
            "tenant_id": tenant_id,
            "project_id": ObjectId(project_id) if isinstance(project_id, str) else project_id,
            "created_at": datetime.utcnow()
        }
        ws_result = await db.workspaces.insert_one(new_ws)
        ws_id = str(ws_result.inserted_id)

        # 2. Forge baseline structural default PRD anchor file matching requirements maps
        initial_prd_template = f"""
        <div class="prd-document-wrapper">
            <h1>📋 PRD: {project_name}</h1>
            <p>Welcome to your automated core system requirements blueprint template.</p>
            <hr style="opacity: 0.2; margin: 16px 0;" />
            <h3>1. Functional Requirements</h3>
            <p>Describe feature specifications parameters rules here...</p>
        </div>
        """
        
        await db.documents.insert_one({
            "workspace_id": ws_id,
            "tenant_id": tenant_id,
            "title": "PRD Default Document",
            "body": initial_prd_template,
            "updated_at": datetime.utcnow(),
            "chat_history": []  # Empty tracking lane allocation matrix
        })

        # 3. AUTOMATION HOOK: Provision empty UI Architecture Canvas
        empty_layout = {"project": project_name, "pages": []}
        await db.canvases.insert_one({
            "project_id": str(project_id),
            "tenant_id": tenant_id,
            "layout_json": empty_layout,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

        return {"workspace_id": ws_id}

    @staticmethod
    async def get_all_workspaces(tenant_id: str) -> List[Dict[str, Any]]:
        """
        Fetches all workspaces for a given tenant. Eagerly populates matching documents.
        """
        db = get_db()
        workspaces_cursor = db.workspaces.find({"tenant_id": tenant_id})
        workspaces = await workspaces_cursor.to_list(length=100)
        
        for ws in workspaces:
            ws_id = str(ws["_id"])
            docs_cursor = db.documents.find({"workspace_id": ws_id, "tenant_id": tenant_id})
            ws["documents"] = await docs_cursor.to_list(length=500)
            
        return workspaces

    @staticmethod
    async def create_workspace(tenant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        db = get_db()
        new_ws = {
            "name": payload["name"],
            "tenant_id": tenant_id,
            "created_at": datetime.utcnow()
        }
        result = await db.workspaces.insert_one(new_ws)
        new_ws["_id"] = result.inserted_id
        return new_ws

    @staticmethod
    async def create_document(tenant_id: str, workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        db = get_db()
        new_doc = {
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "title": payload["title"],
            "body": payload.get("body", ""),
            "updated_at": datetime.utcnow(),
            "chat_history": []
        }
        result = await db.documents.insert_one(new_doc)
        new_doc["_id"] = result.inserted_id
        return new_doc

    @staticmethod
    async def update_document(
        document_id: str, 
        payload: Dict[str, Any], 
        agent_chat_msg: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handles inline character pushes, while atomically appending timestamped 
        chat logs and historical snapshots inside the specific targeted record document file.
        """
        db = get_db()
        if not ObjectId.is_valid(document_id):
            return None
            
        now = datetime.utcnow()
        
        # Build standard attribute setters map
        update_fields = {k: v for k, v in payload.items() if k in ["title", "body"]}
        update_fields["updated_at"] = now
        
        update_query = {"$set": update_fields}

        #  Single File Version Tracker Matrix
        if agent_chat_msg or "body" in payload:
            history_snapshot = {
                "timestamp": now.isoformat(),
                "agent_prompt_or_chat": agent_chat_msg or "Manual content synchronized inline inside editor console workspace.",
                "saved_snapshot_body": payload.get("body", "") or (await db.documents.find_one({"_id": ObjectId(document_id)}))["body"]
            }
            update_query["$push"] = {"chat_history": history_snapshot}

        updated_doc = await db.documents.find_one_and_update(
            {"_id": ObjectId(document_id)},
            update_query,
            return_document=True
        )
        return updated_doc

    @staticmethod
    async def delete_document(document_id: str) -> bool:
        db = get_db()
        if not ObjectId.is_valid(document_id):
            return False
        result = await db.documents.delete_one({"_id": ObjectId(document_id)})
        return result.deleted_count > 0

    @staticmethod
    async def delete_workspace(tenant_id: str, workspace_id: str) -> bool:
        """
        Purges a workspace and executes an atomic cascade deletion 
        on all child PRD documents bound to it.
        """
        db = get_db()
        try:
            ws_oid = ObjectId(workspace_id)
            
            # 1. Cascade Delete: Purge all documents tracking this workspace ID
            doc_purge_result = await db.documents.delete_many({
                "tenant_id": tenant_id,
                "workspace_id": workspace_id # Handles string or OID depending on your schema saving formats
            })
            logger.info(f"Cascade purge removed {doc_purge_result.deleted_count} child documents for workspace {workspace_id}")

            # 2. Drop the parent workspace record row
            ws_purge_result = await db.workspaces.delete_one({
                "_id": ws_oid,
                "tenant_id": tenant_id
            })
            
            return ws_purge_result.deleted_count > 0
            
        except Exception as err:
            logger.error(f"Failed execution on cascade workspace deletion loop: {str(err)}")
            return False

    @staticmethod
    async def get_document_by_id(document_id: str):
        """Fetches a single document record from MongoDB by its structural hex key."""
        db = get_db()
        try:
            doc = await db.documents.find_one({"_id": ObjectId(document_id)})
            return doc
        except Exception:
            return None

    @staticmethod
    async def delete_workspace_async_worker(tenant_id: str, project_id: str):
        db = get_db()
        try:
            project_oid = ObjectId(project_id)
            tenant_oid = ObjectId(tenant_id)
            
            # 1. Find all workspaces associated with this project_id
            workspaces_cursor = db.workspaces.find({
                "$or": [
                    {"tenant_id": tenant_oid},
                    {"tenant_id": tenant_id}
                ],
                "$or": [
                    {"project_id": project_oid},
                    {"project_id": project_id}
                ]
            })
            workspaces = await workspaces_cursor.to_list(length=100)
            
            # 2. Delete all documents in these workspaces, then delete workspaces
            for ws in workspaces:
                ws_id = str(ws["_id"])
                await db.documents.delete_many({
                    "workspace_id": ws_id
                })
                await db.workspaces.delete_one({"_id": ws["_id"]})
                logger.info(f"Asynchronously deleted workspace {ws_id} and its documents for project {project_id}")

            # 3. Delete tasks and sprints associated with project_id
            task_delete_result = await db.tasks.delete_many({
                "project_id": project_oid
            })
            sprint_delete_result = await db.sprints.delete_many({
                "project_id": project_oid
            })
            logger.info(f"Asynchronously deleted tasks ({task_delete_result.deleted_count}) and sprints ({sprint_delete_result.deleted_count}) for project {project_id}")

            # 4. Delete agent workflows, canvases, and generated milestone assets associated with project_id
            canvas_delete_result = await db.canvases.delete_many({
                "$or": [
                    {"project_id": project_id},
                    {"project_id": str(project_id)}
                ]
            })

            wf_cursor = db.project_workflows.find({
                "$or": [
                    {"project_id": project_oid},
                    {"project_id": project_id},
                    {"project_id": str(project_id)}
                ]
            })
            project_workflows = await wf_cursor.to_list(length=100)
            wf_deleted_count = 0
            for pw in project_workflows:
                pw_id = pw["_id"]
                await db.workflow_epics.delete_many({"workflow_id": pw_id})
                await db.workflow_features.delete_many({"workflow_id": pw_id})
                await db.workflow_tasks.delete_many({"workflow_id": pw_id})
                await db.workflow_sprints.delete_many({"workflow_id": pw_id})
                await db.workflow_phases.delete_many({"workflow_id": pw_id})
                await db.workflow_task_dependencies.delete_many({"workflow_id": pw_id})
                await db.project_workflows.delete_one({"_id": pw_id})
                wf_deleted_count += 1

            # Legacy workflows table
            legacy_wf_delete = await db.workflows.delete_many({
                "project_id": project_id
            })

            logger.info(f"Asynchronously deleted project workflows ({wf_deleted_count}), legacy workflows ({legacy_wf_delete.deleted_count}), and canvases ({canvas_delete_result.deleted_count}) for project {project_id}")

        except Exception as err:
            logger.error(f"Error executing asynchronous workspace / project cascade deletion: {str(err)}")

    #  FIX LINE 175: Ensure this decorator aligns EXACTLY with the one above it (4 spaces)
    @staticmethod
    async def get_document_by_id(document_id: str):
        """Fetches a single document record from MongoDB by its structural hex key."""
        db = get_db()
        try:
            doc = await db.documents.find_one({"_id": ObjectId(document_id)})
            return doc
        except Exception:
            return None