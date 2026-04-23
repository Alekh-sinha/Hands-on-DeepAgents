"""Session-scoped backends for deepagents.

Provides two backends:

- ``SessionFilesystemBackend``: file operations sandboxed to a per-session
  directory (no shell execution).
- ``SessionLocalShellBackend``: file operations + local shell execution, both
  scoped to the session directory.  Use this when skills need to run scripts
  (e.g. pptxgenjs, markitdown, Python scripts).
"""

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse


def _npm_global_root() -> str | None:
    """Return the npm global node_modules path, or None if npm is unavailable.

    On Windows `npm` is a .cmd script so shell=True is required.
    """
    try:
        result = subprocess.run(
            "npm root -g",
            capture_output=True, text=True, timeout=10, check=False,
            shell=True,  # required on Windows where npm is npm.cmd
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


def _setup_session_dir(
    source_dirs: list[str | Path] | None,
    sessions_base: str | Path | None,
    session_id: str,
) -> Path:
    """Create and populate the session directory."""
    base = Path(sessions_base).resolve() if sessions_base else Path.cwd() / "sessions"
    session_dir = base / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if source_dirs:
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


class SessionFilesystemBackend(FilesystemBackend):
    """Sandboxed file-only backend scoped to a per-session directory.

    Copies ``source_dirs`` into a fresh session folder and restricts all
    agent file access to that folder via ``virtual_mode=True``.  No shell
    execution is available — use ``SessionLocalShellBackend`` when skills
    need to run scripts.

    Args:
        source_dirs: Files/directories to copy into the sandbox.
        sessions_base: Parent directory for session folders.  Defaults to
            ``./sessions/`` relative to cwd.
        session_id: Explicit session ID.  A random UUID is generated when omitted.
        max_file_size_mb: Forwarded to ``FilesystemBackend``.
    """

    def __init__(
        self,
        source_dirs: list[str | Path] | None = None,
        *,
        sessions_base: str | Path | None = None,
        session_id: str | None = None,
        max_file_size_mb: int = 10,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.session_dir: Path = _setup_session_dir(source_dirs, sessions_base, self.session_id)
        super().__init__(
            root_dir=self.session_dir,
            virtual_mode=True,
            max_file_size_mb=max_file_size_mb,
        )


class SessionLocalShellBackend(LocalShellBackend):
    """Session-scoped backend with both sandboxed file ops and local shell execution.

    Copies ``source_dirs`` into a fresh session folder.  File operations
    (read/write/edit/glob/grep/ls) are restricted to the session folder via
    ``virtual_mode=True``.  Shell commands executed via ``execute()`` run with
    the session folder as the working directory and inherit the full host
    environment so that globally installed tools (node, python, markitdown,
    pptxgenjs, …) are accessible.

    Use this backend when the skills folder contains scripts that need to be
    executed (e.g. the pptx skill which uses pptxgenjs and Python scripts).

    Args:
        source_dirs: Files/directories to copy into the sandbox.
        sessions_base: Parent directory for session folders.  Defaults to
            ``./sessions/`` relative to cwd.
        session_id: Explicit session ID.  A random UUID is generated when omitted.
        timeout: Shell command timeout in seconds (default 300).
        extra_env: Additional environment variables to set for shell commands.
    """

    def __init__(
        self,
        source_dirs: list[str | Path] | None = None,
        *,
        sessions_base: str | Path | None = None,
        session_id: str | None = None,
        timeout: int = 300,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.session_dir: Path = _setup_session_dir(source_dirs, sessions_base, self.session_id)

        # Build env: start from caller extras, then inject NODE_PATH so that
        # `require('pptxgenjs')` works from scripts run inside the session dir.
        env = dict(extra_env or {})
        npm_root = _npm_global_root()
        if npm_root:
            existing_node_path = os.environ.get("NODE_PATH", "")
            env["NODE_PATH"] = (
                f"{npm_root}{os.pathsep}{existing_node_path}"
                if existing_node_path
                else npm_root
            )

        super().__init__(
            root_dir=self.session_dir,
            virtual_mode=True,
            timeout=timeout,
            inherit_env=True,       # host tools (node, python, etc.) must be reachable
            env=env,
        )

    # ------------------------------------------------------------------
    # Virtual-path translation
    # ------------------------------------------------------------------

    def _translate_virtual_paths(self, command: str) -> str:
        """Rewrite /virtual/paths in shell commands to real session_dir paths.

        The agent addresses files with virtual absolute paths (e.g. ``/ml.js``)
        that the file-tools (read/write/edit) resolve correctly via
        ``virtual_mode=True``.  But ``execute()`` passes the command string
        directly to the OS shell, where ``/ml.js`` resolves to the filesystem
        root — not to ``session_dir/ml.js``.

        This scanner replaces every ``/path`` token (outside of shell quotes)
        with the real ``session_dir / rel_path`` when the resolved path exists
        on disk, leaving everything else (option flags, URLs, …) unchanged.
        """
        result: list[str] = []
        j = 0
        n = len(command)
        in_quote: str | None = None

        while j < n:
            ch = command[j]

            # Track quoted regions — leave their content untouched
            if ch in ('"', "'"):
                if in_quote is None:
                    in_quote = ch
                elif in_quote == ch:
                    in_quote = None
                result.append(ch)
                j += 1
                continue

            # Only translate outside of quotes
            if in_quote is None and ch == "/" and j + 1 < n and command[j + 1].isalpha():
                # Scan to the end of the path token
                end = j + 1
                while end < n and command[end] not in (' ', '"', "'", ';', '&', '|', '\n', '\t', ')'):
                    end += 1

                virtual = command[j:end]
                real = self.session_dir / virtual.lstrip("/")
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

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command, translating virtual paths to real paths first."""
        translated = self._translate_virtual_paths(command)
        return super().execute(translated, timeout=timeout)
