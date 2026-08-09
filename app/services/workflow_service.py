from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class WorkflowService:
    @staticmethod
    async def create_workflow(tenant_id: str, payload: dict):
        db = get_db()
        workflow = {
            "tenant_id": tenant_id,
            "project_id": payload["project_id"],
            "nodes": payload.get("nodes", []),
            "edges": payload.get("edges", []),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.workflows.insert_one(workflow)
        workflow["_id"] = result.inserted_id
        return workflow

    @staticmethod
    async def get_workflow_by_project(project_id: str):
        db = get_db()
        return await db.workflows.find_one({"project_id": project_id})

    @staticmethod
    async def update_workflow(project_id: str, update_data: dict):
        db = get_db()
        update_data["updated_at"] = datetime.utcnow()
        return await db.workflows.find_one_and_update(
            {"project_id": project_id},
            {"$set": update_data},
            return_document=True
        )

    @staticmethod
    async def delete_workflow(project_id: str):
        db = get_db()
        await db.workflows.delete_many({"project_id": project_id})
