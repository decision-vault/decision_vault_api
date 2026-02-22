from datetime import datetime

from pydantic import BaseModel, Field


class ChannelCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)


class ThreadCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=120)


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None


class ChannelOut(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime
    updated_at: datetime
    thread_count: int = 0
    message_count: int = 0
    is_favorite: bool = False


class ThreadOut(BaseModel):
    id: str
    channel_id: str
    title: str
    slug: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageOut(BaseModel):
    id: str
    channel_id: str
    thread_id: str | None = None
    content: str
    created_by: str
    created_by_name: str
    created_at: datetime
