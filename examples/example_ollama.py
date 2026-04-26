"""
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
"""

from pathlib import Path

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Model ─────────────────────────────────────────────────────────────────────
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

backend = SessionLocalShellBackend(root_dir=PROJECT_DIR)

# ── Agent ─────────────────────────────────────────────────────────────────────
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
    checkpointer=checkpointer,
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
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if content:
            print(content)
