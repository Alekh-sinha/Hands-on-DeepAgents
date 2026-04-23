# Deep Agent Chat

A Streamlit chat interface for building and running AI agents that can read, write, and execute files — powered by the **[deepagents](https://github.com/langchain-ai/deepagents)** library from LangChain.

Users upload a **skills folder** (containing `SKILL.md` files and any supporting scripts or reference documents), choose a model and paste their API key, then chat with an agent that automatically discovers and applies the uploaded skills.

---

## What is deepagents?

**deepagents** is a LangChain library for creating agentic AI assistants with:

- A structured **Skills system** — the agent reads `SKILL.md` files to learn specialised workflows (e.g. creating presentations, doing research, writing code)
- Built-in **filesystem tools** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`
- **Shell execution** via `execute` — lets the agent run scripts, install packages, and call CLI tools
- **Human-in-the-loop** support via LangGraph checkpointers
- Pluggable **backends** that control where files live and how they are accessed

This project extends deepagents with a **session-scoped sandboxed backend** so that each browser session gets an isolated directory — the agent can only touch files within that sandbox.

---

## Features

- **Per-session file sandbox** — uploaded files are copied into an isolated `sessions/<id>/` directory; the agent cannot read or write anything outside it
- **Skills auto-discovery** — any sub-folder inside the uploaded directory that contains a `SKILL.md` is automatically loaded and injected into the agent's system prompt
- **Cross-references work** — if `SKILL.md` links to `editing.md` or `pptxgenjs.md`, the agent reads those through the same sandboxed backend
- **Shell execution** — the agent can run `node`, `python`, `markitdown`, and any other CLI tool available on the host (useful for skills that generate files)
- **Virtual path translation** — the agent addresses files with virtual absolute paths (`/output.pptx`); the backend transparently maps these to the real session directory before passing commands to the shell
- **No shared secrets** — each user provides their own API key in the UI; it is stored only in their browser session and injected directly into the model constructor (never written to disk or set as a global environment variable)
- **Model dropdown** — supports Google Gemini, Anthropic Claude, OpenAI, Groq, and local Ollama models

---

## Architecture

```
Browser tab
  └─ Streamlit UI (app.py)
       ├─ Sidebar: provider/model dropdown, API key, file uploader
       └─ Chat area

  On "Start Agent":
    1. init_chat_model(model_id, model_provider, api_key=<user key>)
         -> BaseChatModel instance (key embedded, no env var set)

    2. SessionLocalShellBackend(source_dirs=[uploaded files])
         -> copies files into  sessions/<uuid>/
         -> virtual_mode=True  (blocks path traversal)
         -> inherit_env=True   (node, python, etc. reachable)
         -> NODE_PATH injected (npm global modules reachable)

    3. create_deep_agent(model=llm, backend=backend, skills=["/"])
         -> SkillsMiddleware scans /  -> finds sub-dirs with SKILL.md
         -> FilesystemMiddleware wires read/write/edit/ls/glob/grep/execute
         -> MemorySaver checkpointer  (conversation history per session)

  Each chat turn:
    agent.invoke(messages, config={"thread_id": session_id})
```

---

## Supported Models

| Provider | Example models | Requires |
|---|---|---|
| Google Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` | Google API Key |
| Anthropic Claude | `claude-sonnet-4-6`, `claude-opus-4-7` | Anthropic API Key |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o3-mini` | OpenAI API Key |
| Groq | `llama-3.3-70b-versatile` | Groq API Key |
| Ollama | `llama3.2`, `mistral` | Local Ollama server |

Only Google Gemini and Anthropic Claude integration packages are installed by default (see `pyproject.toml`). Install additional provider packages as needed:

```bash
uv add langchain-openai      # OpenAI / Azure OpenAI
uv add langchain-groq        # Groq
uv add langchain-ollama      # Ollama (local)
```

---

## Installation

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd skills

# Create venv and install all dependencies
uv sync
```

### Optional: Node.js tools

If you plan to use skills that generate PowerPoint files via [PptxGenJS](https://gitbrent.github.io/PptxGenJS/):

```bash
npm install -g pptxgenjs
```

---

## Usage

```bash
uv run streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Steps

1. **Select a provider** from the dropdown (Google Gemini, Anthropic Claude, etc.)
2. **Choose a model** from that provider's list
3. **Paste your API key** for the selected provider
4. **Upload your skills folder** — select all files from a folder that contains one or more `SKILL.md` files. Supporting scripts, reference `.md` files, and code are all accessible to the agent.
5. Click **Start / Restart Agent**
6. Chat — ask the agent to do something that the skill handles

### Skills folder format

```
my-skill/
├── SKILL.md          <- required: YAML frontmatter + instructions
├── reference.md      <- optional: the agent reads this when SKILL.md links to it
└── scripts/
    └── helper.py     <- optional: the agent can execute these
```

`SKILL.md` must have YAML frontmatter with at least `name` and `description`:

```markdown
---
name: my-skill
description: "What this skill does and when to use it."
---

# My Skill

## Workflow
...
```

Multiple skill folders can be uploaded at once — the agent discovers all `SKILL.md` files automatically.

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — model selection, file upload, chat loop |
| `session_backend.py` | `SessionLocalShellBackend` and `SessionFilesystemBackend` — sandboxed backends built on top of deepagents' `LocalShellBackend` and `FilesystemBackend` |
| `pyproject.toml` | Package dependencies (managed by uv) |
| `uv.lock` | Pinned dependency versions |

---

## Security notes

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

## License

MIT
