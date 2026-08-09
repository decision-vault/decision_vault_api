from __future__ import annotations

import difflib
import fnmatch
import os
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException


IGNORED_DIRECTORIES = {
    ".cache",
    ".git",
    ".tmp",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
}

TEXT_EXTENSIONS = {
    ".css",
    ".env",
    ".go",
    ".gradle",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".md",
    ".php",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

LANGUAGE_BY_EXTENSION = {
    ".go": "Go",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "React",
    ".kt": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rs": "Rust",
    ".ts": "TypeScript",
    ".tsx": "React",
    ".vue": "Vue",
}


class FilesystemService:
    @staticmethod
    def normalize_root(project_root: str) -> Path:
        root = Path(project_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=400, detail="Project root is not a directory")
        return root

    @staticmethod
    def resolve_path(project_root: str, relative_path: str | None = None) -> Path:
        root = FilesystemService.normalize_root(project_root)
        target = root if not relative_path else (root / relative_path).resolve()
        if root != target and root not in target.parents:
            raise HTTPException(status_code=400, detail="Path escapes project root")
        return target

    @staticmethod
    def read_directory(project_root: str, relative_path: str | None = None) -> list[dict]:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        items = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in IGNORED_DIRECTORIES:
                continue
            items.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(FilesystemService.normalize_root(project_root))),
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return items

    @staticmethod
    def read_file(project_root: str, relative_path: str) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        if target.suffix and target.suffix.lower() not in TEXT_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported binary or unknown file type")
        return {
            "path": relative_path,
            "content": target.read_text(encoding="utf-8", errors="replace"),
            "size": target.stat().st_size,
        }

    @staticmethod
    def write_file(project_root: str, relative_path: str, content: str, confirmed: bool) -> dict:
        if not confirmed:
            return FilesystemService.preview_write_file(project_root, relative_path, content)

        target = FilesystemService.resolve_path(project_root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        target.write_text(content, encoding="utf-8")
        return {
            "status": "written",
            "path": relative_path,
            "diff": FilesystemService.diff_text(relative_path, before, content),
        }

    @staticmethod
    def preview_write_file(project_root: str, relative_path: str, content: str) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        before = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        return {
            "status": "confirmation_required",
            "path": relative_path,
            "operation": "modify" if target.exists() else "create",
            "diff": FilesystemService.diff_text(relative_path, before, content),
        }

    @staticmethod
    def create_file(project_root: str, relative_path: str, content: str, confirmed: bool) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if target.exists():
            raise HTTPException(status_code=409, detail="File already exists")
        return FilesystemService.write_file(project_root, relative_path, content, confirmed)

    @staticmethod
    def delete_file(project_root: str, relative_path: str, confirmed: bool) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        before = target.read_text(encoding="utf-8", errors="replace")
        diff = FilesystemService.diff_text(relative_path, before, "")
        if not confirmed:
            return {"status": "confirmation_required", "path": relative_path, "operation": "delete", "diff": diff}
        target.unlink()
        return {"status": "deleted", "path": relative_path, "diff": diff}

    @staticmethod
    def move_file(project_root: str, from_path: str, to_path: str, confirmed: bool) -> dict:
        source = FilesystemService.resolve_path(project_root, from_path)
        target = FilesystemService.resolve_path(project_root, to_path)
        if not source.exists():
            raise HTTPException(status_code=404, detail="Source not found")
        if not confirmed:
            return {
                "status": "confirmation_required",
                "operation": "move",
                "from_path": from_path,
                "to_path": to_path,
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return {"status": "moved", "from_path": from_path, "to_path": to_path}

    @staticmethod
    def rename_file(project_root: str, relative_path: str, new_name: str, confirmed: bool) -> dict:
        return FilesystemService.move_file(project_root, relative_path, str(Path(relative_path).with_name(new_name)), confirmed)

    @staticmethod
    def create_folder(project_root: str, relative_path: str, confirmed: bool) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if not confirmed:
            return {"status": "confirmation_required", "operation": "create_folder", "path": relative_path}
        target.mkdir(parents=True, exist_ok=True)
        return {"status": "created", "path": relative_path}

    @staticmethod
    def delete_folder(project_root: str, relative_path: str, confirmed: bool) -> dict:
        target = FilesystemService.resolve_path(project_root, relative_path)
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")
        if not confirmed:
            return {"status": "confirmation_required", "operation": "delete_folder", "path": relative_path}
        target.rmdir()
        return {"status": "deleted", "path": relative_path}

    @staticmethod
    def search_files(project_root: str, pattern: str) -> list[dict]:
        root = FilesystemService.normalize_root(project_root)
        matches = []
        for path in FilesystemService.walk_project(root):
            if fnmatch.fnmatch(path.name.lower(), pattern.lower()) or pattern.lower() in path.name.lower():
                matches.append({"name": path.name, "path": str(path.relative_to(root)), "type": "file" if path.is_file() else "directory"})
        return matches[:200]

    @staticmethod
    def search_content(project_root: str, query: str) -> list[dict]:
        root = FilesystemService.normalize_root(project_root)
        results = []
        for path in FilesystemService.walk_project(root):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    if query.lower() in line.lower():
                        results.append({"path": str(path.relative_to(root)), "line": index, "preview": line.strip()[:240]})
                        break
            except OSError:
                continue
        return results[:200]

    @staticmethod
    def index_project(project_root: str) -> dict:
        root = FilesystemService.normalize_root(project_root)
        files = [path for path in FilesystemService.walk_project(root) if path.is_file()]
        languages = FilesystemService.detect_languages(files)
        dependencies = FilesystemService.detect_dependencies(root)
        frameworks = FilesystemService.detect_frameworks(root, dependencies)
        git_repository = (root / ".git").exists()

        return {
            "workspace_id": FilesystemService.workspace_id(root),
            "project_root": str(root),
            "project_name": root.name,
            "language": languages[0] if languages else "Unknown",
            "languages": languages,
            "framework": frameworks[0] if frameworks else "Unknown",
            "frameworks": frameworks,
            "git_repository": git_repository,
            "dependencies": dependencies,
            "package_managers": FilesystemService.detect_package_managers(root),
            "build_system": FilesystemService.detect_build_system(root),
            "docker": (root / "Dockerfile").exists() or (root / "docker-compose.yml").exists(),
            "terraform": any(path.suffix == ".tf" for path in files),
            "tree": FilesystemService.project_tree(root),
            "technology_stack": FilesystemService.technology_stack(languages, frameworks, dependencies),
            "architecture_summary": FilesystemService.architecture_summary(root, files, frameworks),
        }

    @staticmethod
    def walk_project(root: Path) -> Iterable[Path]:
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in IGNORED_DIRECTORIES]
            current_path = Path(current)
            for directory in dirs:
                yield current_path / directory
            for file_name in files:
                yield current_path / file_name

    @staticmethod
    def detect_languages(files: list[Path]) -> list[str]:
        counts: dict[str, int] = {}
        for path in files:
            language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
            if language:
                counts[language] = counts.get(language, 0) + 1
        return [name for name, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]

    @staticmethod
    def detect_dependencies(root: Path) -> list[str]:
        dependencies: set[str] = set()
        package_json = root / "package.json"
        if package_json.exists():
            import json

            try:
                package = json.loads(package_json.read_text(encoding="utf-8"))
                for section in ("dependencies", "devDependencies"):
                    dependencies.update((package.get(section) or {}).keys())
            except Exception:
                pass
        for filename in ("requirements.txt", "requirements.cloud.txt"):
            path = root / filename
            if path.exists():
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    name = line.strip().split("==")[0].split(">=")[0]
                    if name and not name.startswith("#"):
                        dependencies.add(name)
        return sorted(dependencies)[:120]

    @staticmethod
    def detect_frameworks(root: Path, dependencies: list[str]) -> list[str]:
        deps = {dep.lower() for dep in dependencies}
        frameworks = []
        checks = [
            ("Next.js", "next" in deps or (root / "next.config.js").exists()),
            ("React", "react" in deps),
            ("Vue", "vue" in deps),
            ("Angular", "@angular/core" in deps),
            ("FastAPI", "fastapi" in deps),
            ("Express", "express" in deps),
            ("NestJS", "@nestjs/core" in deps),
            ("Spring", (root / "pom.xml").exists() or (root / "build.gradle").exists()),
            ("Laravel", (root / "artisan").exists()),
            ("Django", "django" in deps),
            ("Flutter", (root / "pubspec.yaml").exists()),
            ("React Native", "react-native" in deps),
        ]
        for name, enabled in checks:
            if enabled:
                frameworks.append(name)
        if (root / "Cargo.toml").exists():
            frameworks.append("Rust")
        if (root / "go.mod").exists():
            frameworks.append("Go")
        return frameworks

    @staticmethod
    def detect_package_managers(root: Path) -> list[str]:
        markers = {
            "npm": "package-lock.json",
            "pnpm": "pnpm-lock.yaml",
            "yarn": "yarn.lock",
            "bun": "bun.lockb",
            "poetry": "poetry.lock",
            "uv": "uv.lock",
            "pip": "requirements.txt",
            "cargo": "Cargo.lock",
            "go": "go.mod",
            "maven": "pom.xml",
            "gradle": "build.gradle",
        }
        return [name for name, marker in markers.items() if (root / marker).exists()]

    @staticmethod
    def detect_build_system(root: Path) -> list[str]:
        markers = {
            "Vite": "vite.config.js",
            "Docker": "Dockerfile",
            "Make": "Makefile",
            "Terraform": "main.tf",
            "Gradle": "build.gradle",
            "Maven": "pom.xml",
        }
        return [name for name, marker in markers.items() if (root / marker).exists()]

    @staticmethod
    def project_tree(root: Path, limit: int = 260) -> list[dict]:
        tree = []
        for path in FilesystemService.walk_project(root):
            if len(tree) >= limit:
                break
            tree.append(
                {
                    "path": str(path.relative_to(root)),
                    "name": path.name,
                    "type": "directory" if path.is_dir() else "file",
                    "size": path.stat().st_size if path.is_file() else None,
                }
            )
        return tree

    @staticmethod
    def technology_stack(languages: list[str], frameworks: list[str], dependencies: list[str]) -> list[str]:
        return [*languages[:5], *frameworks[:6], *dependencies[:12]]

    @staticmethod
    def architecture_summary(root: Path, files: list[Path], frameworks: list[str]) -> str:
        top_dirs = [path.name for path in root.iterdir() if path.is_dir() and path.name not in IGNORED_DIRECTORIES]
        parts = [
            f"{root.name} contains {len(files)} indexed files",
            f"top-level areas: {', '.join(top_dirs[:8]) or 'none detected'}",
        ]
        if frameworks:
            parts.append(f"detected frameworks: {', '.join(frameworks)}")
        return ". ".join(parts) + "."

    @staticmethod
    def diff_text(relative_path: str, before: str, after: str) -> str:
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )

    @staticmethod
    def workspace_id(root: Path) -> str:
        import hashlib

        return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
