"""
OpenAI-compatible providers: Google Gemini, Anthropic Claude, OpenAI GPT.

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
"""

from pathlib import Path

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Choose ONE block below — comment out the others ───────────────────────────

# ── Option A: Google Gemini ───────────────────────────────────────────────────
llm = init_chat_model(
    "google_genai:gemini-2.5-flash",    # swap model string for any future Gemini
    api_key="XXXX_GOOGLE_API_KEY_XXXX",
    temperature=0.7,
    max_tokens=8192,                    # alias for max_output_tokens
    top_p=0.95,
    top_k=40,
)

# ── Option B: Anthropic Claude ────────────────────────────────────────────────
# llm = init_chat_model(
#     "anthropic:claude-sonnet-4-6",    # swap for claude-opus-5 etc. when released
#     api_key="XXXX_ANTHROPIC_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,                  # alias for max_tokens_to_sample
#     top_p=0.95,
#     top_k=40,
# )

# ── Option C: OpenAI (current and future models) ──────────────────────────────
# Using "openai:model" is safe for gpt-5 and beyond.
# Auto-inference works for gpt-* and o1/o3 prefixes, but NOT yet for o4+.
# The explicit "openai:" prefix always works regardless of model name pattern.
#
# llm = init_chat_model(
#     "openai:gpt-4o",                  # or "openai:gpt-5", "openai:o4-mini", etc.
#     api_key="XXXX_OPENAI_API_KEY_XXXX",
#     temperature=0.7,
#     max_tokens=8192,
#     top_p=0.95,
#     # Optional: custom base URL for OpenAI-compatible endpoints (Groq, Together, etc.)
#     # base_url="https://api.groq.com/openai/v1",
# )

# ── Backend ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/path/to/your/project")   # ← replace: agent is sandboxed here

backend = SessionLocalShellBackend(root_dir=PROJECT_DIR)

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
    checkpointer=MemorySaver(),
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
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        if content:
            print(content)
