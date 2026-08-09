import asyncio
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth import get_current_user
from app.services.filesystem_service import FilesystemService
from app.services.terminal_service import TerminalService


router = APIRouter(
    prefix="/api/local-workspace",
    tags=["Local Development Workspace"],
    dependencies=[Depends(get_current_user)],
)


class ProjectRootPayload(BaseModel):
    project_root: str


class FilePayload(ProjectRootPayload):
    path: str


class WriteFilePayload(FilePayload):
    content: str
    confirmed: bool = False


class MovePayload(ProjectRootPayload):
    from_path: str
    to_path: str
    confirmed: bool = False


class RenamePayload(FilePayload):
    new_name: str
    confirmed: bool = False


class SearchPayload(ProjectRootPayload):
    query: str = Field(min_length=1)


class CommandPayload(ProjectRootPayload):
    command: str
    env: dict[str, str] | None = None
    timeout_seconds: int = 120


@router.post("/open")
async def open_project(payload: ProjectRootPayload):
    return FilesystemService.index_project(payload.project_root)


@router.post("/index")
async def index_project(payload: ProjectRootPayload):
    return FilesystemService.index_project(payload.project_root)


@router.post("/directory")
async def read_directory(payload: FilePayload):
    return {"items": FilesystemService.read_directory(payload.project_root, payload.path)}


@router.post("/file/read")
async def read_file(payload: FilePayload):
    return FilesystemService.read_file(payload.project_root, payload.path)


@router.post("/file/write")
async def write_file(payload: WriteFilePayload):
    return FilesystemService.write_file(payload.project_root, payload.path, payload.content, payload.confirmed)


@router.post("/file/create")
async def create_file(payload: WriteFilePayload):
    return FilesystemService.create_file(payload.project_root, payload.path, payload.content, payload.confirmed)


@router.post("/file/delete")
async def delete_file(payload: FilePayload):
    return FilesystemService.delete_file(payload.project_root, payload.path, confirmed=False)


@router.post("/file/delete/confirm")
async def confirm_delete_file(payload: FilePayload):
    return FilesystemService.delete_file(payload.project_root, payload.path, confirmed=True)


@router.post("/file/move")
async def move_file(payload: MovePayload):
    return FilesystemService.move_file(payload.project_root, payload.from_path, payload.to_path, payload.confirmed)


@router.post("/file/rename")
async def rename_file(payload: RenamePayload):
    return FilesystemService.rename_file(payload.project_root, payload.path, payload.new_name, payload.confirmed)


@router.post("/folder/create")
async def create_folder(payload: FilePayload):
    return FilesystemService.create_folder(payload.project_root, payload.path, confirmed=False)


@router.post("/folder/create/confirm")
async def confirm_create_folder(payload: FilePayload):
    return FilesystemService.create_folder(payload.project_root, payload.path, confirmed=True)


@router.post("/folder/delete")
async def delete_folder(payload: FilePayload):
    return FilesystemService.delete_folder(payload.project_root, payload.path, confirmed=False)


@router.post("/folder/delete/confirm")
async def confirm_delete_folder(payload: FilePayload):
    return FilesystemService.delete_folder(payload.project_root, payload.path, confirmed=True)


@router.post("/search/files")
async def search_files(payload: SearchPayload):
    return {"results": FilesystemService.search_files(payload.project_root, payload.query)}


@router.post("/search/content")
async def search_content(payload: SearchPayload):
    return {"results": FilesystemService.search_content(payload.project_root, payload.query)}


@router.post("/terminal/run")
async def run_terminal_command(payload: CommandPayload):
    return await TerminalService.run_command(
        payload.project_root,
        payload.command,
        env=payload.env,
        timeout_seconds=payload.timeout_seconds,
    )


@router.post("/terminal/history")
async def terminal_history(payload: ProjectRootPayload):
    return {"history": TerminalService.history(payload.project_root)}


@router.post("/git/status")
async def git_status(payload: ProjectRootPayload):
    root = FilesystemService.normalize_root(payload.project_root)
    if not (root / ".git").exists():
        raise HTTPException(status_code=400, detail="Project is not a Git repository")

    branch, status, commits = await asyncio.gather(
        TerminalService.run_command(payload.project_root, "git branch --show-current", timeout_seconds=20),
        TerminalService.run_command(payload.project_root, "git status --short", timeout_seconds=20),
        TerminalService.run_command(payload.project_root, "git log --oneline -5", timeout_seconds=20),
    )
    return {
        "current_branch": branch["stdout"].strip(),
        "status": status["stdout"].splitlines(),
        "commits": commits["stdout"].splitlines(),
    }
