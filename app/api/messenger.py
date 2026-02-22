from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.middleware.guard import withGuard
from app.schemas.messenger import (
    ChannelCreate,
    ChannelOut,
    MessageCreate,
    MessageOut,
    ThreadCreate,
    ThreadOut,
)
from app.services.messenger_service import (
    create_channel,
    create_message,
    create_thread,
    list_channels,
    list_messages,
    list_threads,
    set_channel_favorite,
)


router = APIRouter(prefix="/api/projects/{project_id}/messenger", tags=["messenger"])


@router.get("/channels", response_model=list[ChannelOut])
async def list_channels_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    return await list_channels(request.state.tenant_id, project_id, user.get("user_id"))


@router.post("/channels", response_model=ChannelOut)
async def create_channel_route(
    project_id: str,
    payload: ChannelCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="member")),
):
    return await create_channel(
        request.state.tenant_id,
        project_id,
        user.get("user_id"),
        payload.name,
    )


@router.get("/channels/{channel_id}/threads", response_model=list[ThreadOut])
async def list_threads_route(
    project_id: str,
    channel_id: str,
    request: Request,
    _user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    return await list_threads(request.state.tenant_id, project_id, channel_id)


@router.post("/channels/{channel_id}/threads", response_model=ThreadOut)
async def create_thread_route(
    project_id: str,
    channel_id: str,
    payload: ThreadCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="member")),
):
    try:
        return await create_thread(
            request.state.tenant_id,
            project_id,
            channel_id,
            user.get("user_id"),
            payload.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/channels/{channel_id}/messages", response_model=list[MessageOut])
async def list_messages_route(
    project_id: str,
    channel_id: str,
    request: Request,
    thread_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    _user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    return await list_messages(
        request.state.tenant_id,
        project_id,
        channel_id,
        thread_id=thread_id,
        limit=limit,
    )


@router.post("/channels/{channel_id}/messages", response_model=MessageOut)
async def create_message_route(
    project_id: str,
    channel_id: str,
    payload: MessageCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="member")),
):
    try:
        return await create_message(
            request.state.tenant_id,
            project_id,
            channel_id,
            user.get("user_id"),
            payload.content,
            payload.thread_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/channels/{channel_id}/favorite")
async def favorite_channel_route(
    project_id: str,
    channel_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="member")),
):
    ok = await set_channel_favorite(
        request.state.tenant_id,
        project_id,
        channel_id,
        user.get("user_id"),
        True,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"status": "favorited"}


@router.delete("/channels/{channel_id}/favorite")
async def unfavorite_channel_route(
    project_id: str,
    channel_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="member")),
):
    ok = await set_channel_favorite(
        request.state.tenant_id,
        project_id,
        channel_id,
        user.get("user_id"),
        False,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"status": "unfavorited"}
