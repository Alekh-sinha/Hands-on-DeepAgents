"""
OpenAI-compatible providers: Google Gemini, Anthropic Claude, OpenAI GPT.

All credentials are passed directly to the model constructor.
Nothing is read from environment variables — replace every XXXX placeholder
with your real value before running.

Requirements
------------
    uv add langchain-google-genai   # for Gemini
    uv add langchain-anthropic      # for Claude
    uv add langchain-openai         # for GPT
"""

from pathlib import Path

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Choose ONE of the three model definitions below ──────────────────────────

# ── Option A: Google Gemini ───────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",           # or "gemini-2.5-pro", "gemini-2.0-flash"
    google_api_key="XXXX_GOOGLE_API_KEY_XXXX",
    temperature=0.7,
    max_output_tokens=8192,
    top_p=0.95,
    top_k=40,
)

# ── Option B: Anthropic Claude ────────────────────────────────────────────────
# from langchain_anthropic import ChatAnthropic
#
# llm = ChatAnthropic(
#     model="claude-sonnet-4-6",          # or "claude-opus-4-7", "claude-haiku-4-5-20251001"
#     api_key="XXXX_ANTHROPIC_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,
#     top_p=0.95,
#     top_k=40,
# )

# ── Option C: OpenAI GPT ──────────────────────────────────────────────────────
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(
#     model="gpt-4o",                     # or "gpt-4o-mini", "o3-mini"
#     api_key="XXXX_OPENAI_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,
#     top_p=0.95,
# )

# ── Backend: agent is sandboxed to this folder ────────────────────────────────
#
# Point root_dir at the folder that contains your skill sub-folders.
# The agent can read/write/execute ONLY inside this directory.
#
# Expected layout:
#   /path/to/project/
#       my-skill/
#           SKILL.md
#           reference.md
#           scripts/
#               helper.py

PROJECT_DIR = Path("/path/to/your/project")   # ← replace with real path

backend = SessionLocalShellBackend(root_dir=PROJECT_DIR)

# ── Agent ─────────────────────────────────────────────────────────────────────
#
# Pass skills=None and use RecursiveSkillsMiddleware so SKILL.md files at any
# nesting depth are discovered automatically.

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,                        # disable the built-in one-level discovery
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
    checkpointer=checkpointer,
)

# ── Invoke ────────────────────────────────────────────────────────────────────
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What skills do you have available?"}]},
    config={"configurable": {"thread_id": "session-001"}},
)

for msg in result["messages"]:
    if msg.__class__.__name__ == "AIMessage":
        content = msg.content
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if content:
            print(content)
