from datetime import datetime
from bson import ObjectId
from fastapi import HTTPException
from app.db.mongo import get_db

class SprintService:
    @staticmethod
    async def create_sprint(tenant_id: str, payload: dict):
        db = get_db()
        sprint = {
            "tenant_id": ObjectId(tenant_id),
            "project_id": ObjectId(payload["project_id"]),
            "name": payload["name"],
            "description": payload.get("description"),
            "start_date": payload["start_date"],
            "end_date": payload["end_date"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "deleted_at": None
        }
        result = await db.sprints.insert_one(sprint)
        sprint["_id"] = result.inserted_id
        return sprint

    @staticmethod
    async def list_sprints(tenant_id: str, project_id: str):
        db = get_db()
        return await db.sprints.find({
            "tenant_id": ObjectId(tenant_id),
            "project_id": ObjectId(project_id),
            "deleted_at": None
        }).to_list(100)

    @staticmethod
    async def get_sprint(sprint_id: str):
        db = get_db()
        return await db.sprints.find_one({"_id": ObjectId(sprint_id), "deleted_at": None})

    @staticmethod
    async def update_sprint(sprint_id: str, update_data: dict):
        db = get_db()
        update_data["updated_at"] = datetime.utcnow()
        return await db.sprints.find_one_and_update(
            {"_id": ObjectId(sprint_id), "deleted_at": None},
            {"$set": update_data},
            return_document=True
        )

    @staticmethod
    async def delete_sprint(sprint_id: str):
        db = get_db()
        await db.sprints.update_one(
            {"_id": ObjectId(sprint_id)},
            {"$set": {"deleted_at": datetime.utcnow()}}
        )
        await db.tasks.update_many(
            {"sprint_id": ObjectId(sprint_id)},
            {"$set": {"sprint_id": None, "updated_at": datetime.utcnow()}}
        )


class TaskService:
    @staticmethod
    async def create_task(tenant_id: str, user_id: str, payload: dict):
        db = get_db()
        sprint_id = None
        parent_id = None
        
        # Validate Sprint if provided
        if payload.get("sprint_id"):
            sprint_id = ObjectId(payload["sprint_id"])
            sprint = await db.sprints.find_one({"_id": sprint_id, "deleted_at": None})
            if not sprint:
                raise HTTPException(status_code=400, detail="Target sprint does not exist.")

        #  Validate Parent Task if this is a Subtask
        if payload.get("parent_id"):
            parent_id = ObjectId(payload["parent_id"])
            parent_task = await db.tasks.find_one({"_id": parent_id, "deleted_at": None})
            if not parent_task:
                raise HTTPException(status_code=400, detail="Parent task does not exist.")

        task = {
            "tenant_id": ObjectId(tenant_id),
            "project_id": ObjectId(payload["project_id"]),
            "sprint_id": sprint_id,
            "parent_id": parent_id, #  Stored reference
            "title": payload["title"],
            "description": payload.get("description"),
            "role": payload["role"],
            "priority": payload["priority"],
            "story_points": payload["story_points"],
            "status": "backlog" if not sprint_id else "ready",
            "assignee_id": None,
            "created_by": ObjectId(user_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "deleted_at": None,
        }

        result = await db.tasks.insert_one(task)
        task["_id"] = result.inserted_id
        return task

    @staticmethod
    async def get_task(task_id: str):
        db = get_db()
        return await db.tasks.find_one({"_id": ObjectId(task_id), "deleted_at": None})

    @staticmethod
    async def list_tasks(project_id: str):
        db = get_db()
        #  Updated to only return root-level tasks by default (parent_id: None)
        return await db.tasks.find({
            "project_id": ObjectId(project_id), 
            "parent_id": None, 
            "deleted_at": None
        }).to_list(100)

    @staticmethod
    async def list_tasks_by_sprint(sprint_id: str):
        db = get_db()
        return await db.tasks.find({
            "sprint_id": ObjectId(sprint_id), 
            "parent_id": None, 
            "deleted_at": None
        }).to_list(100)

    #  NEW: Fetch all subtasks belonging to a specific parent task
    @staticmethod
    async def list_subtasks(parent_id: str):
        db = get_db()
        return await db.tasks.find({
            "parent_id": ObjectId(parent_id), 
            "deleted_at": None
        }).to_list(100)

    @staticmethod
    async def update_task(task_id: str, update_data: dict):
        db = get_db()
        
        if "sprint_id" in update_data:
            sprint_id = update_data["sprint_id"]
            if sprint_id:
                sprint_id = ObjectId(sprint_id)
                sprint = await db.sprints.find_one({"_id": sprint_id, "deleted_at": None})
                if not sprint:
                    raise HTTPException(status_code=400, detail="Target sprint does not exist.")
                update_data["sprint_id"] = sprint_id
            else:
                update_data["sprint_id"] = None

        #  Handle parent_id modifications if updated inline
        if "parent_id" in update_data:
            parent_id = update_data["parent_id"]
            update_data["parent_id"] = ObjectId(parent_id) if parent_id else None

        update_data["updated_at"] = datetime.utcnow()
        return await db.tasks.find_one_and_update(
            {"_id": ObjectId(task_id), "deleted_at": None},
            {"$set": update_data},
            return_document=True
        )

    @staticmethod
    async def delete_task(task_id: str):
        db = get_db()
        now = datetime.utcnow()
        # Soft delete the parent task
        await db.tasks.update_one({"_id": ObjectId(task_id)}, {"$set": {"deleted_at": now}})
        #  Cascade soft-delete: Mark all child subtasks as deleted too
        await db.tasks.update_many({"parent_id": ObjectId(task_id)}, {"$set": {"deleted_at": now}})

    @staticmethod
    async def assign_task(task_id: str, user_id: str):
        db = get_db()
        return await db.tasks.find_one_and_update(
            {"_id": ObjectId(task_id), "deleted_at": None},
            {"$set": {"assignee_id": ObjectId(user_id), "updated_at": datetime.utcnow()}},
            return_document=True
        )

    @staticmethod
    async def unassign_task(task_id: str):
        db = get_db()
        return await db.tasks.find_one_and_update(
            {"_id": ObjectId(task_id), "deleted_at": None},
            {"$set": {"assignee_id": None, "updated_at": datetime.utcnow()}},
            return_document=True
        )

    @staticmethod
    async def get_task_comments(task_id: str):
        db = get_db()
        return await db.comments.find({"task_id": ObjectId(task_id), "deleted_at": None}).sort("created_at", 1).to_list(100)

    @staticmethod
    async def create_task_comment(task_id: str, user_id: str, message: str):
        db = get_db()
        comment = {
            "task_id": ObjectId(task_id),
            "user_id": ObjectId(user_id),
            "message": message,
            "created_at": datetime.utcnow(),
            "deleted_at": None
        }
        result = await db.comments.insert_one(comment)
        comment["_id"] = result.inserted_id
        return comment

    @staticmethod
    async def get_task_activities(task_id: str):
        db = get_db()
        return await db.activities.find({"task_id": ObjectId(task_id)}).sort("created_at", -1).to_list(100)