"""
Stream the agent and log every state transition.

Uses google_genai:gemini-2.5-flash with the pptx skill folder.
Reads the Google API key from .env (GOOGLE_API_KEY=...).

Run:
    uv run python run_streaming.py

What you'll see
---------------
Each line is prefixed with the node name that produced it:
  [model]  — the LLM responded (tool calls or final answer)
  [tools]  — a tool was executed (read_file, write_file, execute, ...)

State integrity is checked after every step:
  - messages list grows monotonically
  - every tool call has a matching tool result
  - no orphaned ToolMessages
"""

import warnings
import sys
import textwrap
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    sys.exit("ERROR: GOOGLE_API_KEY not set in .env")

PROMPT = "create a ppt on Machine learning"
THREAD_ID = "stream-test-01"

# pptx skill folder: must be a direct sub-folder of PROJECT_ROOT so the
# skills middleware finds it at /pptx/SKILL.md
PROJECT_ROOT = Path(__file__).parent
PPTX_DIR = PROJECT_ROOT / "pptx"

if not PPTX_DIR.exists():
    sys.exit(
        f"ERROR: pptx/ folder not found at {PPTX_DIR}\n"
        "Copy your pptx skills folder here before running."
    )

# ── Model ─────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=8192,
)

# ── Backend (sandboxed to project root, pptx/ is a sub-folder) ───────────────

backend = SessionLocalShellBackend(root_dir=PROJECT_ROOT)

# ── Agent ─────────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=None,                    # disable built-in one-level discovery
    middleware=[
        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
    ],
    checkpointer=checkpointer,
)

# ── State integrity checker ───────────────────────────────────────────────────

class StateChecker:
    """Accumulates messages across all streaming chunks and checks integrity."""

    def __init__(self):
        self.all_messages: list = []   # grows with every chunk
        self.issues: list[str] = []

    def ingest(self, new_messages: list, step: int) -> list[str]:
        """Add *new_messages* from this step and return any new violations."""
        self.all_messages.extend(new_messages)

        step_issues = []
        # Rebuild the pending tool-call map from the full history each time.
        # This is O(n) but avoids any cross-chunk ordering assumptions.
        pending: dict[str, str] = {}   # tool_call_id -> tool_name
        resolved: set[str] = set()

        for msg in self.all_messages:
            cls = msg.__class__.__name__
            if cls == "AIMessage":
                for tc in getattr(msg, "tool_calls", []):
                    pending[tc["id"]] = tc["name"]
            elif cls == "ToolMessage":
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id is not None:
                    resolved.add(tc_id)

        orphaned_results = resolved - set(pending.keys())
        unresolved_calls = {k: v for k, v in pending.items() if k not in resolved}

        # Orphaned ToolMessage (no matching call in history) — always wrong
        for tc_id in orphaned_results:
            msg = f"step {step}: ToolMessage id={tc_id!r} has no matching AIMessage tool_call"
            if msg not in self.issues:
                step_issues.append(msg)

        # Unresolved calls — only flag if the last message is a final AI answer
        last_cls = self.all_messages[-1].__class__.__name__ if self.all_messages else ""
        if last_cls == "AIMessage" and not getattr(self.all_messages[-1], "tool_calls", []):
            for tc_id, name in unresolved_calls.items():
                msg = f"step {step}: tool call {name!r} id={tc_id!r} never received a result"
                if msg not in self.issues:
                    step_issues.append(msg)

        self.issues.extend(step_issues)
        return step_issues


def fmt_content(content, max_chars=300) -> str:
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("type", "")
                parts.append(t)
            else:
                parts.append(str(p))
        content = " ".join(parts)
    text = str(content).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


# ── Stream ────────────────────────────────────────────────────────────────────

print("=" * 70)
print(f"Prompt : {PROMPT!r}")
print(f"Model  : gemini-2.5-flash")
print(f"Root   : {PROJECT_ROOT}")
print("=" * 70)
print()

checker = StateChecker()
step = 0

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": PROMPT}]},
    config={"configurable": {"thread_id": THREAD_ID}},
    stream_mode="updates",
):
    step += 1

    for node_name, state_update in chunk.items():
        if state_update is None:
            continue
        raw = state_update.get("messages", [])
        # LangGraph can wrap state values in Overwrite / other sentinel types.
        # Normalise to a plain list of messages.
        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
            messages = list(raw)
        else:
            messages = []

        if not messages:
            keys = list(state_update.keys())
            print(f"[{node_name}] (no messages) state keys: {keys}")
            continue

        print(f"[{node_name}]  +{len(messages)} message(s)")

        for msg in messages:
            cls = msg.__class__.__name__

            if cls == "AIMessage":
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        args_preview = str(tc.get("args", {}))
                        if len(args_preview) > 120:
                            args_preview = args_preview[:120] + "…"
                        print(f"  -> tool_call  {tc['name']}({args_preview})")
                else:
                    # Gemini reasoning models return content as a list of blocks;
                    # pull only text-type blocks (skip "thinking" blocks).
                    raw_c = msg.content
                    if isinstance(raw_c, list):
                        text_parts = [
                            p.get("text", "") for p in raw_c
                            if isinstance(p, dict) and p.get("type") in (None, "text")
                        ]
                        display = " ".join(t for t in text_parts if t).strip()
                    else:
                        display = str(raw_c).strip()
                    if not display:
                        display = "(model returned empty text — reasoning-only or rate-limited)"
                    wrapped = textwrap.fill(display, width=68, initial_indent="  ", subsequent_indent="  ")
                    print(f"  [FINAL ANSWER]\n{wrapped}")

            elif cls == "ToolMessage":
                tool_name = getattr(msg, "name", "?")
                content_preview = fmt_content(msg.content, max_chars=200)
                status = "ok" if not content_preview.startswith("Error") else "ERROR"
                print(f"  <- tool_result [{status}] {tool_name}: {content_preview}")

            elif cls == "HumanMessage":
                print(f"  [human] {fmt_content(msg.content)}")

            else:
                print(f"  [{cls}] {fmt_content(getattr(msg, 'content', ''))}")

        # ── State integrity check (accumulates across all chunks) ─────────
        new_issues = checker.ingest(messages, step)
        for issue in new_issues:
            print(f"  !! INTEGRITY: {issue}")

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Steps completed : {step}")

# Check for generated output files
pptx_files = list(PROJECT_ROOT.rglob("*.pptx"))
if pptx_files:
    print(f"Output PPTX     : {[str(f) for f in pptx_files]}")
else:
    print("Output PPTX     : none found")

if checker.issues:
    print(f"\nIntegrity issues ({len(checker.issues)}):")
    for issue in checker.issues:
        print(f"  - {issue}")
else:
    print("State integrity : all checks passed")
print("=" * 70)
print("=" * 70)
