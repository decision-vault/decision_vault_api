from datetime import datetime, timezone

from bson import ObjectId

from app.db.mongo import get_db


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _serialize_channel(
    doc: dict,
    thread_count: int = 0,
    message_count: int = 0,
    is_favorite: bool = False,
) -> dict:
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "slug": doc["slug"],
        "created_at": doc["created_at"],
        "updated_at": doc.get("updated_at") or doc["created_at"],
        "thread_count": thread_count,
        "message_count": message_count,
        "is_favorite": is_favorite,
    }


def _serialize_thread(doc: dict, message_count: int = 0) -> dict:
    return {
        "id": str(doc["_id"]),
        "channel_id": str(doc["channel_id"]),
        "title": doc["title"],
        "slug": doc["slug"],
        "created_at": doc["created_at"],
        "updated_at": doc.get("updated_at") or doc["created_at"],
        "message_count": message_count,
    }


def _serialize_message(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "channel_id": str(doc["channel_id"]),
        "thread_id": str(doc["thread_id"]) if doc.get("thread_id") else None,
        "content": doc["content"],
        "created_by": str(doc["created_by"]),
        "created_by_name": doc.get("created_by_name") or "Unknown",
        "created_at": doc["created_at"],
    }


def _serialize_personal_message(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "chat_id": str(doc["chat_id"]),
        "content": doc["content"],
        "created_by": str(doc["created_by"]),
        "created_by_name": doc.get("created_by_name") or "Unknown",
        "created_at": doc["created_at"],
    }


def _serialize_personal_contact(doc: dict) -> dict:
    return {
        "user_id": str(doc["_id"]),
        "display_name": doc.get("name") or doc.get("email") or "Unknown",
        "email": doc.get("email") or "",
    }


def _personal_chat_key(user_a_id: str, user_b_id: str) -> str:
    user_ids = sorted([user_a_id, user_b_id])
    return f"{user_ids[0]}:{user_ids[1]}"


async def _resolve_user_name(user_id: str) -> str:
    db = get_db()
    user = await db.users.find_one({"_id": _oid(user_id)}, {"name": 1, "email": 1})
    if not user:
        return "Unknown"
    return user.get("name") or user.get("email") or "Unknown"


async def ensure_default_channel(tenant_id: str, project_id: str, user_id: str) -> None:
    db = get_db()
    exists = await db.project_channels.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    )
    if exists:
        return
    await create_channel(tenant_id, project_id, user_id, "General")


async def list_channels(tenant_id: str, project_id: str, user_id: str) -> list[dict]:
    await ensure_default_channel(tenant_id, project_id, user_id)
    db = get_db()
    channels = await db.project_channels.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    ).sort("created_at", 1).to_list(length=200)

    channel_ids = [doc["_id"] for doc in channels]
    if not channel_ids:
        return []

    thread_counts = {
        item["_id"]: item["count"]
        async for item in db.project_threads.aggregate(
            [
                {
                    "$match": {
                        "tenant_id": _oid(tenant_id),
                        "project_id": _oid(project_id),
                        "channel_id": {"$in": channel_ids},
                        "deleted_at": None,
                    }
                },
                {"$group": {"_id": "$channel_id", "count": {"$sum": 1}}},
            ]
        )
    }

    message_counts = {
        item["_id"]: item["count"]
        async for item in db.project_messages.aggregate(
            [
                {
                    "$match": {
                        "tenant_id": _oid(tenant_id),
                        "project_id": _oid(project_id),
                        "channel_id": {"$in": channel_ids},
                    }
                },
                {"$group": {"_id": "$channel_id", "count": {"$sum": 1}}},
            ]
        )
    }
    favorites = {
        item["channel_id"]
        async for item in db.project_channel_favorites.find(
            {
                "tenant_id": _oid(tenant_id),
                "project_id": _oid(project_id),
                "user_id": _oid(user_id),
            },
            {"channel_id": 1},
        )
    }

    return [
        _serialize_channel(
            doc,
            thread_count=thread_counts.get(doc["_id"], 0),
            message_count=message_counts.get(doc["_id"], 0),
            is_favorite=doc["_id"] in favorites,
        )
        for doc in channels
    ]


async def create_channel(tenant_id: str, project_id: str, user_id: str, name: str) -> dict:
    db = get_db()
    base_slug = _slugify(name)
    slug = base_slug
    suffix = 1
    while await db.project_channels.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "slug": slug,
            "deleted_at": None,
        }
    ):
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "name": name.strip(),
        "slug": slug,
        "created_by": _oid(user_id),
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    result = await db.project_channels.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_channel(doc)


async def list_threads(tenant_id: str, project_id: str, channel_id: str) -> list[dict]:
    db = get_db()
    channel_oid = _oid(channel_id)
    threads = await db.project_threads.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "channel_id": channel_oid,
            "deleted_at": None,
        }
    ).sort("created_at", 1).to_list(length=500)

    thread_ids = [doc["_id"] for doc in threads]
    if not thread_ids:
        return []

    message_counts = {
        item["_id"]: item["count"]
        async for item in db.project_messages.aggregate(
            [
                {
                    "$match": {
                        "tenant_id": _oid(tenant_id),
                        "project_id": _oid(project_id),
                        "channel_id": channel_oid,
                        "thread_id": {"$in": thread_ids},
                    }
                },
                {"$group": {"_id": "$thread_id", "count": {"$sum": 1}}},
            ]
        )
    }
    return [_serialize_thread(doc, message_count=message_counts.get(doc["_id"], 0)) for doc in threads]


async def create_thread(tenant_id: str, project_id: str, channel_id: str, user_id: str, title: str) -> dict:
    db = get_db()
    channel = await db.project_channels.find_one(
        {
            "_id": _oid(channel_id),
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    )
    if not channel:
        raise ValueError("Channel not found")

    base_slug = _slugify(title)
    slug = base_slug
    suffix = 1
    while await db.project_threads.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "channel_id": _oid(channel_id),
            "slug": slug,
            "deleted_at": None,
        }
    ):
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "channel_id": _oid(channel_id),
        "title": title.strip(),
        "slug": slug,
        "created_by": _oid(user_id),
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    result = await db.project_threads.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_thread(doc)


async def list_messages(
    tenant_id: str,
    project_id: str,
    channel_id: str,
    thread_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    db = get_db()
    query: dict = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "channel_id": _oid(channel_id),
    }
    query["thread_id"] = _oid(thread_id) if thread_id else None
    cursor = db.project_messages.find(query).sort("created_at", 1).limit(max(1, min(limit, 500)))
    return [_serialize_message(doc) async for doc in cursor]


async def create_message(
    tenant_id: str,
    project_id: str,
    channel_id: str,
    user_id: str,
    content: str,
    thread_id: str | None = None,
) -> dict:
    db = get_db()
    channel = await db.project_channels.find_one(
        {
            "_id": _oid(channel_id),
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    )
    if not channel:
        raise ValueError("Channel not found")

    if thread_id:
        thread = await db.project_threads.find_one(
            {
                "_id": _oid(thread_id),
                "tenant_id": _oid(tenant_id),
                "project_id": _oid(project_id),
                "channel_id": _oid(channel_id),
                "deleted_at": None,
            }
        )
        if not thread:
            raise ValueError("Thread not found")

    user_name = await _resolve_user_name(user_id)
    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "channel_id": _oid(channel_id),
        "thread_id": _oid(thread_id) if thread_id else None,
        "content": content.strip(),
        "created_by": _oid(user_id),
        "created_by_name": user_name,
        "created_at": now,
        "updated_at": now,
    }
    result = await db.project_messages.insert_one(doc)
    doc["_id"] = result.inserted_id

    await db.project_channels.update_one(
        {"_id": _oid(channel_id)},
        {"$set": {"updated_at": now}},
    )
    if thread_id:
        await db.project_threads.update_one(
            {"_id": _oid(thread_id)},
            {"$set": {"updated_at": now}},
        )
    return _serialize_message(doc)


async def set_channel_favorite(
    tenant_id: str,
    project_id: str,
    channel_id: str,
    user_id: str,
    is_favorite: bool,
) -> bool:
    db = get_db()
    channel = await db.project_channels.find_one(
        {
            "_id": _oid(channel_id),
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    )
    if not channel:
        return False

    query = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "channel_id": _oid(channel_id),
        "user_id": _oid(user_id),
    }
    if is_favorite:
        await db.project_channel_favorites.update_one(
            query,
            {"$set": {**query, "updated_at": _utcnow()}},
            upsert=True,
        )
    else:
        await db.project_channel_favorites.delete_one(query)
    return True


async def _is_project_member(tenant_id: str, project_id: str, user_id: str) -> bool:
    db = get_db()
    try:
        user_oid = _oid(user_id)
    except Exception:
        return False
    member = await db.project_members.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": user_oid,
            "deleted_at": None,
        },
        {"_id": 1},
    )
    return bool(member)


async def list_personal_contacts(tenant_id: str, project_id: str, user_id: str) -> list[dict]:
    db = get_db()
    members = await db.project_members.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
            "user_id": {"$ne": _oid(user_id)},
        },
        {"user_id": 1},
    ).to_list(length=500)
    user_ids = [item.get("user_id") for item in members if item.get("user_id")]
    if not user_ids:
        return []
    users = await db.users.find({"_id": {"$in": user_ids}}, {"name": 1, "email": 1}).to_list(length=500)
    contacts = [_serialize_personal_contact(doc) for doc in users]
    contacts.sort(key=lambda item: (item["display_name"] or "").lower())
    return contacts


async def list_personal_chats(tenant_id: str, project_id: str, user_id: str) -> list[dict]:
    db = get_db()
    user_oid = _oid(user_id)
    chats = await db.project_personal_chats.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "participant_ids": user_oid,
            "deleted_at": None,
        }
    ).sort("updated_at", -1).to_list(length=300)
    if not chats:
        return []

    chat_ids = [doc["_id"] for doc in chats]
    message_counts = {
        item["_id"]: item["count"]
        async for item in db.project_personal_messages.aggregate(
            [
                {
                    "$match": {
                        "tenant_id": _oid(tenant_id),
                        "project_id": _oid(project_id),
                        "chat_id": {"$in": chat_ids},
                    }
                },
                {"$group": {"_id": "$chat_id", "count": {"$sum": 1}}},
            ]
        )
    }
    last_messages = {
        item["_id"]: item
        async for item in db.project_personal_messages.aggregate(
            [
                {
                    "$match": {
                        "tenant_id": _oid(tenant_id),
                        "project_id": _oid(project_id),
                        "chat_id": {"$in": chat_ids},
                    }
                },
                {"$sort": {"chat_id": 1, "created_at": -1}},
                {
                    "$group": {
                        "_id": "$chat_id",
                        "content": {"$first": "$content"},
                        "created_at": {"$first": "$created_at"},
                    }
                },
            ]
        )
    }

    other_user_ids = set()
    for chat in chats:
        for participant_id in chat.get("participant_ids") or []:
            if participant_id != user_oid:
                other_user_ids.add(participant_id)
    users = await db.users.find({"_id": {"$in": list(other_user_ids)}}, {"name": 1, "email": 1}).to_list(length=500)
    user_map = {doc["_id"]: doc for doc in users}

    payload = []
    for chat in chats:
        participant_ids = chat.get("participant_ids") or []
        other_user_id = next((pid for pid in participant_ids if pid != user_oid), None)
        profile = user_map.get(other_user_id, {})
        last_message = last_messages.get(chat["_id"])
        preview = None
        if last_message and last_message.get("content"):
            preview = str(last_message["content"])[:120]
        payload.append(
            {
                "id": str(chat["_id"]),
                "participant_user_id": str(other_user_id) if other_user_id else "",
                "participant_display_name": profile.get("name") or profile.get("email") or "Unknown",
                "participant_email": profile.get("email") or "",
                "created_at": chat["created_at"],
                "updated_at": chat.get("updated_at") or chat["created_at"],
                "last_message_preview": preview,
                "last_message_at": last_message.get("created_at") if last_message else None,
                "message_count": message_counts.get(chat["_id"], 0),
            }
        )
    return payload


async def create_personal_chat(
    tenant_id: str,
    project_id: str,
    user_id: str,
    participant_user_id: str,
) -> dict:
    if participant_user_id == user_id:
        raise ValueError("Cannot create a personal chat with yourself")
    db = get_db()
    if not await _is_project_member(tenant_id, project_id, participant_user_id):
        raise ValueError("Participant is not a project member")

    participant_key = _personal_chat_key(user_id, participant_user_id)
    existing = await db.project_personal_chats.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "participant_key": participant_key,
            "deleted_at": None,
        }
    )
    if existing:
        chats = await list_personal_chats(tenant_id, project_id, user_id)
        for chat in chats:
            if chat["id"] == str(existing["_id"]):
                return chat

    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "participant_ids": [_oid(user_id), _oid(participant_user_id)],
        "participant_key": participant_key,
        "created_by": _oid(user_id),
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    result = await db.project_personal_chats.insert_one(doc)
    user = await db.users.find_one({"_id": _oid(participant_user_id)}, {"name": 1, "email": 1})
    return {
        "id": str(result.inserted_id),
        "participant_user_id": participant_user_id,
        "participant_display_name": user.get("name") if user else "Unknown",
        "participant_email": user.get("email") if user else "",
        "created_at": now,
        "updated_at": now,
        "last_message_preview": None,
        "last_message_at": None,
        "message_count": 0,
    }


async def list_personal_messages(
    tenant_id: str,
    project_id: str,
    chat_id: str,
    user_id: str,
    limit: int = 200,
) -> list[dict]:
    db = get_db()
    try:
        chat_oid = _oid(chat_id)
        user_oid = _oid(user_id)
    except Exception as exc:
        raise ValueError("Personal chat not found") from exc
    chat = await db.project_personal_chats.find_one(
        {
            "_id": chat_oid,
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "participant_ids": user_oid,
            "deleted_at": None,
        },
        {"_id": 1},
    )
    if not chat:
        raise ValueError("Personal chat not found")

    cursor = db.project_personal_messages.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "chat_id": chat_oid,
        }
    ).sort("created_at", 1).limit(max(1, min(limit, 500)))
    return [_serialize_personal_message(doc) async for doc in cursor]


async def create_personal_message(
    tenant_id: str,
    project_id: str,
    chat_id: str,
    user_id: str,
    content: str,
) -> dict:
    db = get_db()
    try:
        chat_oid = _oid(chat_id)
        user_oid = _oid(user_id)
    except Exception as exc:
        raise ValueError("Personal chat not found") from exc
    chat = await db.project_personal_chats.find_one(
        {
            "_id": chat_oid,
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "participant_ids": user_oid,
            "deleted_at": None,
        }
    )
    if not chat:
        raise ValueError("Personal chat not found")

    user_name = await _resolve_user_name(user_id)
    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "chat_id": chat_oid,
        "content": content.strip(),
        "created_by": user_oid,
        "created_by_name": user_name,
        "created_at": now,
        "updated_at": now,
    }
    result = await db.project_personal_messages.insert_one(doc)
    doc["_id"] = result.inserted_id
    await db.project_personal_chats.update_one(
        {"_id": chat_oid},
        {"$set": {"updated_at": now}},
    )
    return _serialize_personal_message(doc)
