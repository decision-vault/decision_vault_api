from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class CanvasService:
    @staticmethod
    async def create_canvas(tenant_id: str, payload: dict):
        db = get_db()
        canvas = {
            "tenant_id": tenant_id,
            "project_id": payload["project_id"],
            "layout_json": payload.get("layout_json", {}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.canvases.insert_one(canvas)
        canvas["_id"] = result.inserted_id
        return canvas

    @staticmethod
    async def get_canvas_by_project(project_id: str):
        db = get_db()
        return await db.canvases.find_one({"project_id": project_id})

    @staticmethod
    async def update_canvas(project_id: str, update_data: dict):
        db = get_db()
        update_data["updated_at"] = datetime.utcnow()
        return await db.canvases.find_one_and_update(
            {"project_id": project_id},
            {"$set": update_data},
            return_document=True
        )

    @staticmethod
    async def delete_canvas(project_id: str):
        db = get_db()
        await db.canvases.delete_many({"project_id": project_id})
