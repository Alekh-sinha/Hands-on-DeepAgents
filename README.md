# Skills Agent

A production-ready web UI for running AI agents powered by [langchain-deepagents](https://pypi.org/project/langchain-deepagents/).  
Agents discover skills from SKILL.md files, execute shell commands inside a sandboxed session directory, and stream responses in real time.

## Features

| Feature | Details |
|---------|---------|
| **Login & sessions** | Every login creates an isolated session directory; agents are sandboxed inside it |
| **Skills panel** | Automatically discovers all SKILL.md files at any folder depth and displays them as tiles |
| **Model registry** | Register any provider (Gemini, Claude, OpenAI, Azure, Groq, Mistral, Ollama) and switch on the fly |
| **Streaming chat** | Token-by-token SSE streaming with live tool-call and tool-result indicators |
| **Skill upload** | Upload a ZIP of a skills folder; agent reloads automatically on the next message |
| **AI Training tab** | Placeholder for future fine-tuning and dataset management features |

---

## Architecture

```
skills/
├── api/
│   └── main.py          ← FastAPI backend (auth, skills, models, chat SSE)
├── web/
│   ├── index.html       ← Login page
│   ├── app.html         ← Main SPA (chat + skills + model registry)
│   └── static/
│       ├── main.css
│       └── app.js
├── session_backend.py   ← Sandboxed backends + RecursiveSkillsMiddleware
├── cli.py               ← Interactive CLI alternative
├── app.py               ← Streamlit alternative
└── pyproject.toml
```

**Request flow**

```
Browser ──POST /api/chat/stream──► FastAPI
                                       │
                                       ▼
                              create_deep_agent
                              (langchain-deepagents)
                                       │
                              RecursiveSkillsMiddleware
                              discovers SKILL.md files
                                       │
                              SessionLocalShellBackend
                              (sandboxed to sessions/<id>)
                                       │
                              SSE token stream ────────────► Browser
```

---

## Local setup

### 1  Prerequisites

| Tool | Version |
|------|---------|
| Python | >= 3.13 |
| [uv](https://docs.astral.sh/uv/) | any recent |

```bash
# Install uv if you don't have it
pip install uv
```

### 2  Install dependencies

```bash
uv sync
```

### 3  Configure skills template (optional)

Create a folder containing your skill sub-folders and point `SKILLS_TEMPLATE_DIR` at it.
Every new login will get a private copy of all skills inside it.

```
skills_template/
├── my-skill/
│   ├── SKILL.md
│   └── run.py
└── another-skill/
    ├── SKILL.md
    └── script.js
```

```bash
# Windows
set SKILLS_TEMPLATE_DIR=C:\path\to\skills_template

# macOS / Linux
export SKILLS_TEMPLATE_DIR=/path/to/skills_template
```

If `SKILLS_TEMPLATE_DIR` is not set the session directory starts empty (you can still upload a ZIP).

### 4  Start the server

```bash
# Development (auto-reloads only when api/ code changes)
uv run uvicorn api.main:app --reload --reload-dir api --host 0.0.0.0 --port 8000

# Production / normal use (no reload)
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **Why `--reload-dir api`?**  
> `--reload` with no `--reload-dir` watches the *entire* project directory.  
> Every login creates a new `sessions/<uuid>/` folder, which watchfiles detects
> as a file change and immediately reloads the module — wiping all in-memory
> sessions and causing "Invalid or expired session" on the next request.  
> `--reload-dir api` limits watching to `api/main.py` only, so session state
> survives across requests.

Open **http://localhost:8000** in your browser.

### 5  Sign in

Enter any email and password — in local mode all credentials are accepted and create a new session.

### 6  Register a model

Click **+** in the Model Registry panel and fill in:

| Provider | Required fields |
|----------|----------------|
| Google Gemini | API key |
| Anthropic Claude | API key |
| OpenAI | API key |
| Azure OpenAI | API key · endpoint · API version |
| Groq | API key |
| Mistral AI | API key |
| Ollama | Base URL (default: `http://localhost:11434`) |

The first registered model is selected automatically.

### 7  Chat

Type a message and press **Enter** (Shift+Enter for a newline).
The agent streams its response token by token. Tool calls and their results appear inline.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLS_TEMPLATE_DIR` | `skills_template` | Folder copied into every new session |
| `SESSIONS_BASE` | `sessions` | Root directory for session sandboxes |

---

## SKILL.md format

Each skill lives in its own folder with a `SKILL.md` file:

```markdown
---
name: my-skill
description: One-line summary shown in the UI
version: 1.0.0
---

# My Skill

Longer description and usage instructions for the agent...
```

The agent discovers every `SKILL.md` at any depth inside the session directory.

---

## CLI alternative

```bash
uv run python cli.py
```

Interactive terminal chatbot with the same provider support and streaming output.

---

## Streamlit alternative

```bash
uv run streamlit run app.py
```

---

## Security notes

<<<<<<< HEAD
- **API keys** are injected directly into the `BaseChatModel` constructor and held only in Streamlit session state for the duration of the browser session. They are never written to disk.
- **File sandbox** — `virtual_mode=True` on the backend blocks path traversal (`..`, `~`, absolute paths outside the session directory) at the file-tools level. The agent cannot read files outside `sessions/<id>/`.
- **Shell execution** is not sandboxed at the OS level — commands run on the host with the current user's permissions. Only deploy this in trusted environments or with Human-in-the-Loop interrupts enabled.
- Do **not** commit `.env` files — the `.gitignore` excludes them.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `deepagents` | >= 0.5.3 | Agent framework, skills system, filesystem middleware |
| `streamlit` | >= 1.56.0 | Web UI |
| `langchain-google-genai` | (transitive) | Google Gemini integration |
| `langchain-anthropic` | (transitive) | Anthropic Claude integration |
| `langgraph` | (transitive) | Stateful agent graph, checkpointing |
| `markitdown[pptx]` | >= 0.1.5 | Extract text from Office files for content QA |
| `Pillow` | >= 12.2.0 | Image handling (used by some skills) |

---
=======
- Session directories are sandboxed via `virtual_mode=True` — agents cannot read or write outside their assigned folder.
- API keys are stored only in server memory for the lifetime of the session and never written to disk.
- Add proper authentication before exposing this to the public internet.
>>>>>>> deepagents-experiment
