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


# ── Config ─────────────────────────────────────────────────────────────────
SESSIONS_BASE = Path(os.environ.get("SESSIONS_BASE", "sessions"))
SKILLS_TEMPLATE_DIR = Path(os.environ.get("SKILLS_TEMPLATE_DIR", "skills_template"))

# Pre-create sessions dir at import time so watchfiles doesn't count its
# creation as a "file change" and trigger a spurious module reload.
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

        try:
            async for chunk, _meta in agent.astream(
                {"messages": [{"role": "user", "content": body.message}]},
                config={"configurable": {"thread_id": session["session_id"]}},
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
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
