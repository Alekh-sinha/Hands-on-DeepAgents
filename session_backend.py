"""Session-scoped backends and recursive skills middleware for deepagents.

Drop-in replacements for deepagents' built-in backends that enforce
``virtual_mode=True`` so the agent cannot escape the given ``root_dir``.

Classes
-------
SessionFilesystemBackend  – file ops only, sandboxed to root_dir
SessionLocalShellBackend  – file ops + shell execution, sandboxed to root_dir
RecursiveSkillsMiddleware – finds SKILL.md files at *any* nesting depth

Helper
------
make_session_dir(source_dirs, sessions_base, session_id) -> Path
    Copy files into a fresh session directory and return its path.
    Use this in the Streamlit UI when handling uploaded files.

Usage
-----
# Point at an existing project folder (nothing is copied):
backend = SessionLocalShellBackend(root_dir="/path/to/project")
agent = create_deep_agent(model=llm, backend=backend, skills=["/"])

# Streamlit upload flow:
session_dir = make_session_dir(source_dirs=[uploaded_tmp])
backend = SessionLocalShellBackend(root_dir=session_dir)
agent = create_deep_agent(model=llm, backend=backend, skills=["/"],
                          middleware=[RecursiveSkillsMiddleware(backend=backend, sources=["/"])])
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.middleware.skills import (
    SkillsMiddleware,
    SkillsStateUpdate,
    _parse_skill_metadata,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime
    from langchain_core.runnables import RunnableConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _npm_global_root() -> str | None:
    """Return the npm global node_modules path; None if npm is unavailable.

    Uses shell=True because on Windows npm is a .cmd batch file.
    """
    try:
        r = subprocess.run(
            "npm root -g",
            capture_output=True, text=True, timeout=10, check=False, shell=True,
        )
        path = r.stdout.strip()
        return path if path else None
    except Exception:
        return None


def make_session_dir(
    source_dirs: list[str | Path],
    sessions_base: str | Path | None = None,
    session_id: str | None = None,
) -> Path:
    """Copy *source_dirs* into a fresh session directory and return its path.

    Intended for the Streamlit UI upload flow where files come from a temp
    directory.  Each directory in *source_dirs* is copied as a named
    sub-folder so the session root can hold several skill sets at once.

    Example::

        tmp = save_uploads_to_tempdir(uploaded_files)
        session_dir = make_session_dir(source_dirs=[tmp])
        backend = SessionLocalShellBackend(root_dir=session_dir)
    """
    sid = session_id or str(uuid.uuid4())
    base = Path(sessions_base).resolve() if sessions_base else Path.cwd() / "sessions"
    session_dir = base / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dirs:
        src = Path(src).resolve()
        if not src.exists():
            continue
        if src.is_dir():
            dest = session_dir / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, session_dir / src.name)

    return session_dir


# ---------------------------------------------------------------------------
# SessionFilesystemBackend
# ---------------------------------------------------------------------------

class SessionFilesystemBackend(FilesystemBackend):
    """Sandboxed drop-in replacement for FilesystemBackend.

    Forces ``virtual_mode=True`` so the agent cannot read or write anything
    outside *root_dir*.  Path traversal (``..``, ``~``, absolute paths
    outside the root) raises ``ValueError`` before reaching the filesystem.

    Swap one line to harden an existing agent::

        # Before (full filesystem access):
        backend = FilesystemBackend(root_dir="/path/to/project")

        # After (sandboxed to that folder):
        backend = SessionFilesystemBackend(root_dir="/path/to/project")
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        max_file_size_mb: int = 10,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            virtual_mode=True,
            max_file_size_mb=max_file_size_mb,
        )


# ---------------------------------------------------------------------------
# SessionLocalShellBackend
# ---------------------------------------------------------------------------

class SessionLocalShellBackend(LocalShellBackend):
    """Sandboxed drop-in replacement for LocalShellBackend.

    Extends ``LocalShellBackend`` with three additions:

    1. **virtual_mode=True** – file tools (read/write/edit/ls/glob/grep) are
       restricted to *root_dir*; the agent cannot escape via ``..`` or
       absolute paths.

    2. **NODE_PATH injection** – automatically detects the npm global
       node_modules directory so that ``require('pptxgenjs')`` (and other
       globally installed npm packages) works from scripts inside the
       sandbox without a local ``node_modules/``.

    3. **Virtual path translation** – ``execute()`` receives commands with
       virtual absolute paths (e.g. ``node /ml.js``) but the OS shell
       resolves ``/ml.js`` to the filesystem root, not *root_dir*.  The
       ``execute()`` override rewrites any ``/virtual/path`` token whose
       real counterpart (``root_dir / rel_path``) exists on disk before
       the command is handed to the shell.

    Swap one line::

        # Before:
        backend = LocalShellBackend(root_dir="/path/to/project")

        # After (sandboxed + NODE_PATH + path translation):
        backend = SessionLocalShellBackend(root_dir="/path/to/project")
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> None:
        merged_env = dict(env or {})

        # Inject NODE_PATH so globally-installed npm packages are require()-able
        # from scripts inside the sandbox directory.
        npm_root = _npm_global_root()
        if npm_root:
            existing = os.environ.get("NODE_PATH", "")
            merged_env["NODE_PATH"] = (
                f"{npm_root}{os.pathsep}{existing}" if existing else npm_root
            )

        super().__init__(
            root_dir=root_dir,
            virtual_mode=True,
            timeout=timeout,
            inherit_env=inherit_env,
            env=merged_env,
        )

    # ------------------------------------------------------------------
    # Virtual-path translation
    # ------------------------------------------------------------------

    def _translate_virtual_paths(self, command: str) -> str:
        """Rewrite ``/virtual/path`` tokens in shell commands to real paths.

        The agent addresses files with virtual absolute paths (``/ml.js``)
        which the file-tools resolve correctly via ``virtual_mode=True``.
        But ``execute()`` passes the string straight to the OS shell, where
        ``/ml.js`` resolves to the filesystem root, not to ``root_dir/ml.js``.

        This scanner replaces every unquoted ``/path`` token with
        ``root_dir / rel_path`` when that real path exists on disk.
        URLs, option flags (``--flag``), and paths that don't exist are left
        untouched.
        """
        result: list[str] = []
        j, n = 0, len(command)
        in_quote: str | None = None

        while j < n:
            ch = command[j]

            # Track quoted regions — never translate inside quotes
            if ch in ('"', "'"):
                in_quote = None if in_quote == ch else (in_quote or ch)
                result.append(ch)
                j += 1
                continue

            if in_quote is None and ch == "/" and j + 1 < n and command[j + 1].isalpha():
                # Scan to end of the path token
                end = j + 1
                while end < n and command[end] not in (' ', '"', "'", ';', '&', '|', '\n', '\t', ')'):
                    end += 1

                virtual = command[j:end]
                real = self.cwd / virtual.lstrip("/")
                if real.exists():
                    real_str = str(real)
                    if " " in real_str:
                        real_str = f'"{real_str}"'
                    result.append(real_str)
                else:
                    result.append(virtual)
                j = end
                continue

            result.append(ch)
            j += 1

        return "".join(result)

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Translate virtual paths, then execute."""
        return super().execute(self._translate_virtual_paths(command), timeout=timeout)


# ---------------------------------------------------------------------------
# RecursiveSkillsMiddleware
# ---------------------------------------------------------------------------

class RecursiveSkillsMiddleware(SkillsMiddleware):
    """Skills middleware that discovers SKILL.md files at *any* nesting depth.

    The built-in ``SkillsMiddleware`` only looks **one level deep** — it calls
    ``backend.ls(source_path)`` and checks each direct child directory for a
    ``SKILL.md``.  This means a structure like::

        /skills/
            category-a/
                my-skill/
                    SKILL.md   <- NOT found by standard middleware
            my-other-skill/
                SKILL.md       <- found (direct child)

    ``RecursiveSkillsMiddleware`` uses ``backend.glob("**/SKILL.md")`` to find
    every ``SKILL.md`` at any depth, so *all* skills are loaded regardless of
    how deeply they are nested.

    Usage::

        from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

        backend = SessionLocalShellBackend(root_dir="/path/to/project")
        agent = create_deep_agent(
            model=llm,
            backend=backend,
            skills=None,                        # disable built-in skills discovery
            middleware=[
                RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
            ],
        )

    Note: pass ``skills=None`` to ``create_deep_agent`` to prevent the built-in
    ``SkillsMiddleware`` from also running.
    """

    def _collect_all_skills(self, backend, source_path: str):
        """Glob for every SKILL.md under *source_path* and parse them all."""
        glob_result = backend.glob("**/SKILL.md", path=source_path)
        skill_md_paths = [m["path"] for m in (glob_result.matches or [])]

        if not skill_md_paths:
            return []

        responses = backend.download_files(skill_md_paths)
        skills = []
        for path, resp in zip(skill_md_paths, responses):
            if resp.error or not resp.content:
                continue
            try:
                content = resp.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            # directory_name must match the 'name' field in SKILL.md frontmatter
            directory_name = PurePosixPath(path).parent.name
            metadata = _parse_skill_metadata(content, path, directory_name)
            if metadata:
                skills.append(metadata)
        return skills

    async def _acollect_all_skills(self, backend, source_path: str):
        """Async version of _collect_all_skills."""
        glob_result = await backend.aglob("**/SKILL.md", path=source_path)
        skill_md_paths = [m["path"] for m in (glob_result.matches or [])]

        if not skill_md_paths:
            return []

        responses = await backend.adownload_files(skill_md_paths)
        skills = []
        for path, resp in zip(skill_md_paths, responses):
            if resp.error or not resp.content:
                continue
            try:
                content = resp.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            directory_name = PurePosixPath(path).parent.name
            metadata = _parse_skill_metadata(content, path, directory_name)
            if metadata:
                skills.append(metadata)
        return skills

    def before_agent(self, state, runtime, config=None):
        # config is injected by the framework only for base deepagents classes;
        # we default to None and pass it through — _get_backend ignores it when
        # the backend is a direct instance (not a factory).
        if "skills_metadata" in state:
            return None
        backend = self._get_backend(state, runtime, config)
        all_skills: dict = {}
        for source_path in self.sources:
            for skill in self._collect_all_skills(backend, source_path):
                all_skills[skill["name"]] = skill
        return SkillsStateUpdate(skills_metadata=list(all_skills.values()))

    async def abefore_agent(self, state, runtime, config=None):
        if "skills_metadata" in state:
            return None
        backend = self._get_backend(state, runtime, config)
        all_skills: dict = {}
        for source_path in self.sources:
            for skill in await self._acollect_all_skills(backend, source_path):
                all_skills[skill["name"]] = skill
        return SkillsStateUpdate(skills_metadata=list(all_skills.values()))
