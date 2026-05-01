"""
<<<<<<< HEAD
Local open-source models via Ollama.

No API key needed — Ollama runs entirely on your machine.
The agent is still sandboxed to root_dir via SessionLocalShellBackend.

Setup
-----
1. Install Ollama: https://ollama.com/download
2. Pull a model:
       ollama pull llama3.2        # 3 B — fast, low VRAM
       ollama pull mistral         # 7 B — good balance
       ollama pull llama3.1:70b    # 70 B — best quality, needs ~40 GB RAM
3. Ollama starts automatically; it listens on http://localhost:11434

Requirements
------------
    uv add langchain-ollama
=======
Local open-source models via Ollama — no API key needed.

Uses init_chat_model as the aggregator.

How Ollama maps through init_chat_model
-----------------------------------------
  init_chat_model("ollama:llama3.2", base_url="http://localhost:11434", ...)
  -> _parse_model: provider="ollama", model="llama3.2"
  -> ChatOllama(model="llama3.2", base_url="http://localhost:11434", ...)

Why auto-inference doesn't work for Ollama
-------------------------------------------
_attempt_infer_model_provider() has no rule for Ollama model names.
"llama3.2", "mistral", "phi3" all return None.  Always use "ollama:model" format.

Setup
-----
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:
       ollama pull llama3.2        # 3B — fast, ~2 GB VRAM
       ollama pull mistral         # 7B — good balance, ~5 GB VRAM
       ollama pull llama3.1:70b    # 70B — best quality, ~40 GB VRAM / RAM
  3. Ollama starts a local server automatically on http://localhost:11434

Requirements
------------
  uv add langchain-ollama
>>>>>>> deepagents-experiment
"""

from pathlib import Path

from deepagents import create_deep_agent
<<<<<<< HEAD
from langchain_ollama import ChatOllama
=======
from langchain.chat_models import init_chat_model
>>>>>>> deepagents-experiment
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Model ─────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
llm = ChatOllama(
    model="llama3.2",               # must match the name you pulled with ollama pull
    base_url="http://localhost:11434",  # default; change if Ollama is on another host

    # Generation parameters
    temperature=0.7,
    num_predict=4096,               # max tokens to generate (Ollama's max_tokens)
    top_p=0.9,
    top_k=40,

    # Performance tuning (optional)
    # num_ctx=8192,                 # context window size
    # num_gpu=1,                    # number of GPU layers to offload
)

# ── Backend ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/path/to/your/project")   # ← replace with real path
=======
llm = init_chat_model(
    "ollama:llama3.2",                  # swap for any model you have pulled

    # Ollama server address — change if running on a remote machine
    base_url="http://localhost:11434",

    # Generation parameters
    temperature=0.7,
    num_predict=4096,                   # Ollama's name for max output tokens
    top_p=0.9,
    top_k=40,

    # Optional performance tuning
    # num_ctx=8192,                     # context window size
    # num_gpu=1,                        # GPU layers to offload
    # repeat_penalty=1.1,
)

# ── Backend ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/path/to/your/project")   # ← replace
>>>>>>> deepagents-experiment

backend = SessionLocalShellBackend(root_dir=PROJECT_DIR)

# ── Agent ─────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
checkpointer = MemorySaver()

=======
>>>>>>> deepagents-experiment
agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
<<<<<<< HEAD
    checkpointer=checkpointer,
=======
    checkpointer=MemorySaver(),
>>>>>>> deepagents-experiment
)

# ── Invoke ────────────────────────────────────────────────────────────────────
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What skills do you have available?"}]},
    config={"configurable": {"thread_id": "ollama-session-001"}},
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
