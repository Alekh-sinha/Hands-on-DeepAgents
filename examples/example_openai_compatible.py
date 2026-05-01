"""
OpenAI-compatible providers: Google Gemini, Anthropic Claude, OpenAI GPT.

<<<<<<< HEAD
All credentials are passed directly to the model constructor.
Nothing is read from environment variables — replace every XXXX placeholder
with your real value before running.

Requirements
------------
    uv add langchain-google-genai   # for Gemini
    uv add langchain-anthropic      # for Claude
    uv add langchain-openai         # for GPT
=======
Uses init_chat_model as a single aggregator instead of importing provider-specific
classes (ChatGoogleGenerativeAI, ChatOpenAI, etc.).

Why init_chat_model?
--------------------
- One import, every provider.  No `from langchain_xxx import ChatXxx` per provider.
- Every kwarg is passed verbatim to the underlying class constructor via **kwargs,
  so you can pass any parameter the model supports without changing this file.
- Uses the "provider:model" format which is always explicit — no auto-inference
  guessing.  Works for future models (gpt-5, gemini-3.0, claude-opus-5, ...) without
  any code change: just update the model string.
- api_key is a Pydantic alias accepted by every LangChain provider class, so it
  routes correctly to google_api_key / anthropic_api_key / openai_api_key etc.

How it works under the hood
---------------------------
  init_chat_model("google_genai:gemini-2.5-flash", api_key="xxx", temperature=0.7)
  -> _parse_model("google_genai:gemini-2.5-flash")
        provider = "google_genai",  model = "gemini-2.5-flash"
  -> _get_chat_model_creator("google_genai")
        imports langchain_google_genai.ChatGoogleGenerativeAI
  -> ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="xxx", temperature=0.7)

Requirements
------------
  uv add langchain-google-genai   # for Gemini
  uv add langchain-anthropic      # for Claude
  uv add langchain-openai         # for GPT / o-series
>>>>>>> deepagents-experiment
"""

from pathlib import Path

from deepagents import create_deep_agent
<<<<<<< HEAD
=======
from langchain.chat_models import init_chat_model
>>>>>>> deepagents-experiment
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

<<<<<<< HEAD
# ── Choose ONE of the three model definitions below ──────────────────────────

# ── Option A: Google Gemini ───────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",           # or "gemini-2.5-pro", "gemini-2.0-flash"
    google_api_key="XXXX_GOOGLE_API_KEY_XXXX",
    temperature=0.7,
    max_output_tokens=8192,
=======
# ── Choose ONE block below — comment out the others ───────────────────────────

# ── Option A: Google Gemini ───────────────────────────────────────────────────
llm = init_chat_model(
    "google_genai:gemini-2.5-flash",    # swap model string for any future Gemini
    api_key="XXXX_GOOGLE_API_KEY_XXXX",
    temperature=0.7,
    max_tokens=8192,                    # alias for max_output_tokens
>>>>>>> deepagents-experiment
    top_p=0.95,
    top_k=40,
)

# ── Option B: Anthropic Claude ────────────────────────────────────────────────
<<<<<<< HEAD
# from langchain_anthropic import ChatAnthropic
#
# llm = ChatAnthropic(
#     model="claude-sonnet-4-6",          # or "claude-opus-4-7", "claude-haiku-4-5-20251001"
#     api_key="XXXX_ANTHROPIC_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,
=======
# llm = init_chat_model(
#     "anthropic:claude-sonnet-4-6",    # swap for claude-opus-5 etc. when released
#     api_key="XXXX_ANTHROPIC_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,                  # alias for max_tokens_to_sample
>>>>>>> deepagents-experiment
#     top_p=0.95,
#     top_k=40,
# )

<<<<<<< HEAD
# ── Option C: OpenAI GPT ──────────────────────────────────────────────────────
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(
#     model="gpt-4o",                     # or "gpt-4o-mini", "o3-mini"
=======
# ── Option C: OpenAI (current and future models) ──────────────────────────────
# Using "openai:model" is safe for gpt-5 and beyond.
# Auto-inference works for gpt-* and o1/o3 prefixes, but NOT yet for o4+.
# The explicit "openai:" prefix always works regardless of model name pattern.
#
# llm = init_chat_model(
#     "openai:gpt-4o",                  # or "openai:gpt-5", "openai:o4-mini", etc.
>>>>>>> deepagents-experiment
#     api_key="XXXX_OPENAI_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,
#     top_p=0.95,
<<<<<<< HEAD
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
=======
#     # Optional: custom base URL for OpenAI-compatible endpoints (Groq, Together, etc.)
#     # base_url="https://api.groq.com/openai/v1",
# )

# ── Backend ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/path/to/your/project")   # ← replace: agent is sandboxed here
>>>>>>> deepagents-experiment

backend = SessionLocalShellBackend(root_dir=PROJECT_DIR)

# ── Agent ─────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
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
=======
agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
    checkpointer=MemorySaver(),
>>>>>>> deepagents-experiment
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
<<<<<<< HEAD
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
=======
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
>>>>>>> deepagents-experiment
        if content:
            print(content)
