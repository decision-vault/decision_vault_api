from __future__ import annotations

import asyncio
import shlex
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from app.services.filesystem_service import FilesystemService


SUPPORTED_COMMANDS = {
    "bun",
    "cargo",
    "docker",
    "git",
    "go",
    "gradle",
    "make",
    "mvn",
    "node",
    "npm",
    "pip",
    "pnpm",
    "poetry",
    "pytest",
    "python",
    "python3",
    "terraform",
    "uv",
    "yarn",
}

COMMAND_HISTORY: list[dict] = []


class TerminalService:
    @staticmethod
    async def run_command(project_root: str, command: str, env: dict[str, str] | None = None, timeout_seconds: int = 120) -> dict:
        root = FilesystemService.normalize_root(project_root)
        parts = shlex.split(command)
        if not parts:
            raise HTTPException(status_code=400, detail="Command is required")

        executable = parts[0]
        if executable not in SUPPORTED_COMMANDS:
            raise HTTPException(status_code=400, detail=f"Unsupported command: {executable}")

        started_at = datetime.utcnow()
        process_env = None
        if env:
            import os

            process_env = {**os.environ, **{key: str(value) for key, value in env.items()}}

        process = await asyncio.create_subprocess_exec(
            *parts,
            cwd=str(root),
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise HTTPException(status_code=408, detail="Command timed out and was cancelled")

        entry = {
            "project_root": str(root),
            "command": command,
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
        }
        COMMAND_HISTORY.append(entry)
        del COMMAND_HISTORY[:-100]
        return entry

    @staticmethod
    def history(project_root: str | None = None) -> list[dict]:
        if not project_root:
            return COMMAND_HISTORY[-50:]
        root = str(Path(project_root).expanduser().resolve())
        return [entry for entry in COMMAND_HISTORY if entry["project_root"] == root][-50:]
