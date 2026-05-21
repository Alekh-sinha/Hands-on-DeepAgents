"""FastAPI backend for Skills Agent UI."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import secrets
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import AsyncGenerator

# Allow importing session_backend from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from session_backend import (
    SessionLocalShellBackend,
    RecursiveSkillsMiddleware,
    make_session_dir,
)
from deepagents import create_deep_agent

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

LANGFUSE_ENABLED = False
_LANGFUSE_PUBLIC_KEY: str | None = None

try:
    _pk  = os.environ["LANGFUSE_PUBLIC_KEY"]
    _sk  = os.environ["LANGFUSE_SECRET_KEY"]
    _host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Register credentials into LangfuseResourceManager._instances once.
    Langfuse(public_key=_pk, secret_key=_sk, host=_host)

    # Keep a module-level client for flush() calls.
    _langfuse_client = get_client(public_key=_pk)

    _LANGFUSE_PUBLIC_KEY = _pk
    LANGFUSE_ENABLED = True
    print(f"[langfuse] tracing enabled → {_host}")
except KeyError:
    _langfuse_client = None
    print("[langfuse] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled")
except Exception as exc:
    _langfuse_client = None
    print(f"[langfuse] init failed ({exc}) — tracing disabled")


def _make_langfuse_handler(session_id: str, user_id: str) -> LangfuseCallbackHandler | None:
    """
    Return a per-request CallbackHandler scoped to this session.

    CallbackHandler(public_key=...) does a direct dict lookup in
    LangfuseResourceManager._instances — it does NOT re-read credentials
    or create a new connection. The public_key is only used as the lookup key.

    session_id and user_id are passed via reserved metadata keys at invoke
    time (langfuse_session_id, langfuse_user_id) rather than on the handler
    constructor, which is the v3 pattern.
    """
    if not LANGFUSE_ENABLED:
        return None
    try:
        return LangfuseCallbackHandler(public_key=_LANGFUSE_PUBLIC_KEY)
    except Exception as exc:
        print(f"[langfuse] handler creation failed: {exc}")
        return None


# ── Config ─────────────────────────────────────────────────────────────────
SESSIONS_BASE = Path(os.environ.get("SESSIONS_BASE", "sessions"))
SKILLS_TEMPLATE_DIR = Path(os.environ.get("SKILLS_TEMPLATE_DIR", "skills_template"))

SESSIONS_BASE.mkdir(parents=True, exist_ok=True)

# ── In-memory state ─────────────────────────────────────────────────────────
_tokens: dict[str, str] = {}     # token -> session_id
_sessions: dict[str, dict] = {}  # session_id -> session data

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Skills Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helpers ─────────────────────────────────────────────────────────────
def _make_token(session_id: str) -> str:
    token = secrets.token_urlsafe(32)
    _tokens[token] = session_id
    return token


def _get_session(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:].strip()
    sid = _tokens.get(token)
    if not sid or sid not in _sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return _sessions[sid]


# ── SKILL.md parser ──────────────────────────────────────────────────────────
def _parse_frontmatter(content: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    result: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip().strip("\"'")
    return result


def _scan_skills(session_dir: Path) -> list[dict]:
    skills = []
    for skill_md in sorted(session_dir.rglob("SKILL.md")):
        try:
            content = skill_md.read_text(encoding="utf-8")
            fm = _parse_frontmatter(content)
            if fm.get("name"):
                skills.append(
                    {
                        "name": fm["name"],
                        "description": fm.get("description", ""),
                        "version": fm.get("version", ""),
                        "path": str(skill_md.relative_to(session_dir)).replace("\\", "/"),
                    }
                )
        except Exception:
            pass
    return skills


# ── Agent helpers ─────────────────────────────────────────────────────────────
def _build_agent(session: dict):
    idx = session["active_model_idx"]
    if idx is None or not session["models"]:
        raise HTTPException(
            status_code=400,
            detail="No model selected. Please register and select a model first.",
        )

    m = session["models"][idx]
    model_spec = f"{m['provider']}:{m['model']}"
    kwargs: dict = dict(m.get("extra") or {})
    if m.get("api_key"):
        kwargs["api_key"] = m["api_key"]

    try:
        llm = init_chat_model(model_spec, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model init failed: {exc}")

    session_dir: Path = session["session_dir"]
    backend = SessionLocalShellBackend(root_dir=session_dir)
    checkpointer = session.get("checkpointer") or MemorySaver()

    with contextlib.redirect_stdout(io.StringIO()):
        agent = create_deep_agent(
            model=llm,
            backend=backend,
            skills=None,
            middleware=[RecursiveSkillsMiddleware(backend=backend, sources=["/"])],
            checkpointer=checkpointer,
        )

    session["agent"] = agent
    session["backend"] = backend
    session["checkpointer"] = checkpointer
    return agent


def _ensure_agent(session: dict):
    if session.get("agent") is None:
        return _build_agent(session)
    return session["agent"]


# ── Auth endpoints ────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
def login(body: LoginRequest):
    session_id = str(uuid.uuid4())

    source_dirs = [SKILLS_TEMPLATE_DIR] if SKILLS_TEMPLATE_DIR.exists() else []
    session_dir = make_session_dir(
        source_dirs=source_dirs,
        sessions_base=SESSIONS_BASE,
        session_id=session_id,
    )

    _sessions[session_id] = {
        "session_id": session_id,      # used as LangGraph thread_id (one per session)
        "email": body.email,
        "session_dir": session_dir,
        "models": [],
        "active_model_idx": None,
        "agent": None,
        "backend": None,
        "checkpointer": None,
    }

    return {"token": _make_token(session_id), "email": body.email}


@app.post("/auth/logout")
def logout(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        sid = _tokens.pop(token, None)
        if sid:
            _sessions.pop(sid, None)
    return {"ok": True}


# ── Skills endpoints ──────────────────────────────────────────────────────────
@app.get("/api/skills")
def list_skills(request: Request):
    session = _get_session(request)
    return {"skills": _scan_skills(session["session_dir"])}


@app.post("/api/skills/upload")
async def upload_skills(request: Request, file: UploadFile = File(...)):
    session = _get_session(request)
    session_dir: Path = session["session_dir"]

    tmp_zip = session_dir / f"_up_{uuid.uuid4().hex}.zip"
    extract_dir = session_dir / f"_ex_{uuid.uuid4().hex}"

    try:
        tmp_zip.write_bytes(await file.read())
        extract_dir.mkdir()

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(extract_dir)

        members = [p for p in extract_dir.iterdir()]
        if len(members) == 1 and members[0].is_dir():
            dest = session_dir / members[0].name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(members[0]), str(dest))
        else:
            stem = Path(file.filename or "skills").stem
            dest = session_dir / stem
            dest.mkdir(exist_ok=True)
            for item in members:
                shutil.move(str(item), str(dest / item.name))
    finally:
        tmp_zip.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

    session["agent"] = None
    return {"ok": True, "message": "Skills uploaded. Agent will reload on next message."}


# ── Model registry endpoints ──────────────────────────────────────────────────
class ModelRegisterRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    name: str
    extra: dict | None = None


@app.get("/api/models")
def list_models(request: Request):
    session = _get_session(request)
    return {
        "models": [
            {
                "id": i,
                "provider": m["provider"],
                "model": m["model"],
                "name": m["name"],
                "extra": m.get("extra") or {},
            }
            for i, m in enumerate(session["models"])
        ],
        "active_id": session["active_model_idx"],
    }


@app.post("/api/models")
def register_model(body: ModelRegisterRequest, request: Request):
    session = _get_session(request)
    session["models"].append(
        {
            "provider": body.provider,
            "model": body.model,
            "api_key": body.api_key,
            "name": body.name,
            "extra": body.extra or {},
        }
    )
    idx = len(session["models"]) - 1
    if session["active_model_idx"] is None:
        session["active_model_idx"] = idx
        session["agent"] = None
    return {"id": idx, "name": body.name}


@app.put("/api/models/{model_id}/select")
def select_model(model_id: int, request: Request):
    session = _get_session(request)
    if not (0 <= model_id < len(session["models"])):
        raise HTTPException(status_code=404, detail="Model not found")
    session["active_model_idx"] = model_id
    session["agent"] = None
    return {"ok": True, "active_id": model_id}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: int, request: Request):
    session = _get_session(request)
    if not (0 <= model_id < len(session["models"])):
        raise HTTPException(status_code=404, detail="Model not found")
    session["models"].pop(model_id)
    active = session["active_model_idx"]
    if active is not None:
        if active == model_id:
            session["active_model_idx"] = 0 if session["models"] else None
            session["agent"] = None
        elif active > model_id:
            session["active_model_idx"] = active - 1
    return {"ok": True}


# ── Chat SSE endpoint ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    session = _get_session(request)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            agent = _ensure_agent(session)
        except HTTPException as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': exc.detail})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── Langfuse: per-request handler ──────────────────────────────────
        # _make_langfuse_handler() calls CallbackHandler(public_key=...) which
        # does a dict lookup in LangfuseResourceManager._instances — it does
        # NOT re-read credentials or create a new HTTP connection.
        # Returns None gracefully if Langfuse is not configured.
        langfuse_handler = _make_langfuse_handler(
            session_id=session["session_id"],
            user_id=session["email"],
        )

        # Build the LangGraph config.
        # session_id, user_id, and tags are passed via reserved metadata keys
        # (v3 pattern) — the handler reads these automatically and stamps every
        # trace/span, so there is no need to pass them to the constructor.
        extra_callbacks = [langfuse_handler] if langfuse_handler else []
        config = {
            "configurable": {"thread_id": session["session_id"]},
            "callbacks": extra_callbacks,
            "metadata": {
                "langfuse_session_id": session["session_id"],
                "langfuse_user_id": session["email"],
                "langfuse_tags": ["skills-agent"],
            },
        }

        try:
            async for chunk, _meta in agent.astream(
                {"messages": [{"role": "user", "content": body.message}]},
                config=config,
                stream_mode="messages",
            ):
                cls = chunk.__class__.__name__

                if cls == "AIMessageChunk":
                    content = chunk.content
                    if isinstance(content, str) and content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in (None, "text"):
                                t = part.get("text", "")
                                if t:
                                    yield f"data: {json.dumps({'type': 'token', 'content': t})}\n\n"

                    for tc in chunk.tool_call_chunks or []:
                        if tc.get("name"):
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name']})}\n\n"

                elif cls == "ToolMessage":
                    result = chunk.content
                    if isinstance(result, list):
                        result = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in result
                        )
                    first_line = str(result).split("\n")[0][:300]
                    yield f"data: {json.dumps({'type': 'tool_result', 'content': first_line})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            # Flush this session's Langfuse buffer so spans are not lost if
            # the process idles before the background worker drains naturally.
            if langfuse_handler and _langfuse_client:
                try:
                    _langfuse_client.flush()
                except Exception:
                    pass
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File management helpers ───────────────────────────────────────────────────
BINARY_EXTENSIONS = {
    ".zip", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".webm", ".ogg",
    ".bin", ".pyc", ".exe", ".dll", ".xlsx", ".xls", ".docx", ".doc",
    ".pptx", ".ppt", ".db", ".sqlite",
}

_VERSION_RE = re.compile(r"^.+\.v\d+(\.\w+)?$")


def _safe_path(session_dir: Path, rel_path: str) -> Path:
    rel_path = rel_path.lstrip("/").lstrip("\\")
    if not rel_path:
        return session_dir
    resolved = (session_dir / rel_path).resolve()
    if not str(resolved).startswith(str(session_dir.resolve())):
        raise HTTPException(status_code=400, detail="Path escapes session directory")
    return resolved


def _build_tree(root: Path, session_dir: Path) -> list[dict]:
    entries = []
    try:
        items = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return []
    for item in items:
        name = item.name
        if name.startswith(("_up_", "_ex_", ".")):
            continue
        if item.is_file() and _VERSION_RE.match(name):
            continue
        rel_str = str(item.relative_to(session_dir)).replace("\\", "/")
        if item.is_dir():
            entries.append({
                "name": name,
                "path": rel_str,
                "type": "dir",
                "children": _build_tree(item, session_dir),
            })
        else:
            ext = item.suffix.lower()
            entries.append({
                "name": name,
                "path": rel_str,
                "type": "file",
                "ext": ext,
                "size": item.stat().st_size,
                "is_binary": ext in BINARY_EXTENSIONS,
            })
    return entries


def _file_versions(file_path: Path, session_dir: Path) -> list[dict]:
    stem = file_path.stem
    suffix = file_path.suffix
    pat = re.compile(
        rf"^{re.escape(stem)}\.v(\d+){re.escape(suffix)}$" if suffix
        else rf"^{re.escape(stem)}\.v(\d+)$"
    )
    versions = []
    try:
        for f in file_path.parent.iterdir():
            m = pat.match(f.name)
            if m and f.is_file():
                rel = str(f.relative_to(session_dir)).replace("\\", "/")
                versions.append({
                    "label": f"v{m.group(1)}",
                    "num": int(m.group(1)),
                    "filename": f.name,
                    "rel_path": rel,
                    "size": f.stat().st_size,
                })
    except Exception:
        pass
    versions.sort(key=lambda x: x["num"])
    return versions


def _next_version_name(file_path: Path, session_dir: Path) -> str:
    versions = _file_versions(file_path, session_dir)
    n = (versions[-1]["num"] + 1) if versions else 1
    stem = file_path.stem
    suffix = file_path.suffix
    return f"{stem}.v{n}{suffix}" if suffix else f"{stem}.v{n}"


# ── File management endpoints ─────────────────────────────────────────────────

@app.get("/api/files")
def list_files(request: Request):
    session = _get_session(request)
    tree = _build_tree(session["session_dir"], session["session_dir"])
    return {"tree": tree}


@app.get("/api/files/content")
def read_file_content(request: Request, path: str):
    session = _get_session(request)
    file_path = _safe_path(session["session_dir"], path)
    if not file_path.exists() or file_path.is_dir():
        raise HTTPException(status_code=404, detail="File not found")
    ext = file_path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return {"content": None, "is_binary": True, "size": file_path.stat().st_size, "versions": []}
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    versions = _file_versions(file_path, session["session_dir"])
    return {"content": content, "is_binary": False, "versions": versions}


class SaveFileRequest(BaseModel):
    path: str
    content: str


@app.post("/api/files/save")
def save_file(body: SaveFileRequest, request: Request):
    session = _get_session(request)
    session_dir: Path = session["session_dir"]
    file_path = _safe_path(session_dir, body.path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        backup_name = _next_version_name(file_path, session_dir)
        shutil.copy2(str(file_path), str(file_path.parent / backup_name))
    file_path.write_text(body.content, encoding="utf-8")
    session["agent"] = None
    versions = _file_versions(file_path, session_dir)
    return {"ok": True, "versions": versions}


class RestoreVersionRequest(BaseModel):
    path: str
    version_name: str


@app.post("/api/files/restore")
def restore_version(body: RestoreVersionRequest, request: Request):
    session = _get_session(request)
    session_dir: Path = session["session_dir"]
    file_path = _safe_path(session_dir, body.path)
    version_path = _safe_path(session_dir, str(Path(body.path).parent / body.version_name))
    if not version_path.exists():
        raise HTTPException(status_code=404, detail="Version not found")
    if file_path.exists():
        backup_name = _next_version_name(file_path, session_dir)
        shutil.copy2(str(file_path), str(file_path.parent / backup_name))
    shutil.copy2(str(version_path), str(file_path))
    session["agent"] = None
    return {"ok": True}


@app.delete("/api/files")
def delete_file(request: Request, path: str):
    session = _get_session(request)
    target = _safe_path(session["session_dir"], path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if target.is_dir():
        shutil.rmtree(str(target))
    else:
        target.unlink()
    session["agent"] = None
    return {"ok": True}


class MkdirRequest(BaseModel):
    path: str


@app.post("/api/files/mkdir")
def make_directory(body: MkdirRequest, request: Request):
    session = _get_session(request)
    target = _safe_path(session["session_dir"], body.path)
    target.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@app.post("/api/upload")
async def upload_any_file(request: Request, file: UploadFile = File(...), dest_folder: str = ""):
    session = _get_session(request)
    session_dir: Path = session["session_dir"]
    dest_dir = _safe_path(session_dir, dest_folder) if dest_folder else session_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload").name
    data = await file.read()

    if filename.lower().endswith(".zip"):
        tmp_zip = dest_dir / f"_up_{uuid.uuid4().hex}.zip"
        extract_dir = dest_dir / f"_ex_{uuid.uuid4().hex}"
        try:
            tmp_zip.write_bytes(data)
            extract_dir.mkdir()
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(extract_dir)
            members = list(extract_dir.iterdir())
            stem = Path(filename).stem
            if len(members) == 1 and members[0].is_dir():
                dest = dest_dir / members[0].name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(members[0]), str(dest))
            else:
                dest = dest_dir / stem
                dest.mkdir(exist_ok=True)
                for item in members:
                    shutil.move(str(item), str(dest / item.name))
            session["agent"] = None
            rel_path = str(dest.relative_to(session_dir)).replace("\\", "/")
            return {"ok": True, "path": rel_path, "message": f"Extracted {filename} → {dest.name}/"}
        except Exception as _exc:
            raise HTTPException(status_code=400, detail=f"ZIP extraction failed: {_exc}") from _exc
        finally:
            tmp_zip.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)

    dest_path = dest_dir / filename
    dest_path.write_bytes(data)
    session["agent"] = None
    rel_path = str(dest_path.relative_to(session_dir)).replace("\\", "/")
    return {"ok": True, "path": rel_path, "message": f"Uploaded {filename}"}


@app.get("/api/session/download")
def download_session(request: Request):
    session = _get_session(request)
    session_dir: Path = session["session_dir"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(session_dir.rglob("*")):
            if not fp.is_file():
                continue
            if fp.name.startswith(("_up_", "_ex_")):
                continue
            arcname = str(fp.relative_to(session_dir)).replace("\\", "/")
            zf.write(str(fp), arcname)
    buf.seek(0)
    email_slug = re.sub(r"[^\w]", "_", session["email"].split("@")[0])
    return StreamingResponse(
        io.BytesIO(buf.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="session_{email_slug}.zip"'},
    )


# ── Static file serving ───────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_WEB = _ROOT / "web"

if (_WEB / "static").exists():
    app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(_WEB / "index.html"))


@app.get("/app")
def main_app():
    return FileResponse(str(_WEB / "app.html"))