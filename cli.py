#!/usr/bin/env python3
"""
CLI chatbot powered by deepagents with streaming output.

Usage
-----
  python cli.py                        # interactive setup
  python cli.py --provider google_genai --model gemini-2.5-flash
  python cli.py --provider azure_openai --model my-deployment
  python cli.py --skills /path/to/project   # load skills from a folder

Commands inside the chat
-------------------------
  /quit  /exit     exit
  /clear           reset conversation history
  /skills          show loaded skills
  /help            show this list
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ── ANSI colours (Windows 10+ / all POSIX terminals support these) ──────────
R  = "\033[0m"      # reset
B  = "\033[1m"      # bold
CY = "\033[36m"     # cyan  — assistant text
YL = "\033[33m"     # yellow — tool calls
GR = "\033[90m"     # grey  — tool results
GN = "\033[32m"     # green — status / prompts
RD = "\033[31m"     # red   — errors

# Enable ANSI and UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")
    # Force UTF-8 so box-drawing / emoji chars don't crash on cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Provider catalogue ────────────────────────────────────────────────────────
# Each provider has:
#   params   - fields to collect (in order). `env` = env-var to pre-fill from.
#   optional - kwargs the user can override (shown with defaults, Enter to keep)

PROVIDERS: dict[str, dict[str, Any]] = {
    "google_genai": {
        "label": "Google Gemini",
        "default_model": "gemini-2.5-flash",
        "params": [
            {"name": "api_key",   "env": "GOOGLE_API_KEY",  "label": "Google API Key",  "secret": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 8192},
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "default_model": "claude-sonnet-4-6",
        "params": [
            {"name": "api_key",   "env": "ANTHROPIC_API_KEY", "label": "Anthropic API Key", "secret": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 8192},
    },
    "openai": {
        "label": "OpenAI (GPT / o-series)",
        "default_model": "gpt-4o",
        "params": [
            {"name": "api_key",   "env": "OPENAI_API_KEY",  "label": "OpenAI API Key",   "secret": True},
            {"name": "base_url",  "env": "OPENAI_BASE_URL", "label": "Base URL (Enter = api.openai.com)", "secret": False, "optional": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 8192},
    },
    "azure_openai": {
        "label": "Azure OpenAI (deployed model)",
        "default_model": "",   # deployment name — must be entered
        "params": [
            {"name": "azure_endpoint", "env": "AZURE_OPENAI_ENDPOINT",  "label": "Azure endpoint (https://...openai.azure.com/)", "secret": False},
            {"name": "api_key",        "env": "AZURE_OPENAI_API_KEY",   "label": "Azure API Key",   "secret": True},
            {"name": "api_version",    "env": "OPENAI_API_VERSION",     "label": "API version",     "secret": False, "default": "2025-01-01-preview"},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 4096},
    },
    "ollama": {
        "label": "Ollama (local open-source)",
        "default_model": "llama3.2",
        "params": [
            {"name": "base_url", "env": "OLLAMA_HOST", "label": "Ollama host", "secret": False, "default": "http://localhost:11434"},
        ],
        "optional": {"temperature": 0.7, "num_predict": 4096},
    },
    "groq": {
        "label": "Groq",
        "default_model": "llama-3.3-70b-versatile",
        "params": [
            {"name": "api_key", "env": "GROQ_API_KEY", "label": "Groq API Key", "secret": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 8192},
    },
    "mistralai": {
        "label": "Mistral AI",
        "default_model": "mistral-large-latest",
        "params": [
            {"name": "api_key", "env": "MISTRAL_API_KEY", "label": "Mistral API Key", "secret": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 8192},
    },
    "litellm": {
        "label": "LiteLLM proxy (100+ providers via one endpoint)",
        "default_model": "gpt-4o",
        "params": [
            {"name": "api_base", "env": "LITELLM_API_BASE", "label": "LiteLLM base URL (http://localhost:4000)", "secret": False},
            {"name": "api_key",  "env": "LITELLM_API_KEY",  "label": "LiteLLM API Key (if auth enabled, else Enter)", "secret": True, "optional": True},
        ],
        "optional": {"temperature": 0.7, "max_tokens": 4096},
    },
}

PROVIDER_NAMES = list(PROVIDERS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt the user; return default on empty input."""
    suffix = f" [{default}]" if default else ""
    full_prompt = f"{GN}{prompt}{suffix}: {R}"
    try:
        val = (getpass.getpass(full_prompt) if secret else input(full_prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val or default


def _is_interactive() -> bool:
    """True when stdin is a real terminal (not piped/redirected)."""
    return sys.stdin.isatty()


def _collect_param(p: dict) -> tuple[str, str]:
    """Collect one provider parameter; pre-fill from env var if available."""
    name    = p["name"]
    label   = p["label"]
    secret  = p.get("secret", False)
    env_key = p.get("env", "")
    default = p.get("default", "")

    # Pre-fill from env
    env_val = os.environ.get(env_key, "") if env_key else ""
    if env_val:
        masked = ("*" * 8) if secret else env_val
        if _is_interactive():
            # Real terminal: offer chance to override
            print(f"  {GN}{label}{R}: {masked}  {GR}(from .env — Enter to keep){R}")
            try:
                override = (getpass.getpass("  > ") if secret else input("  > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            return name, override or env_val
        else:
            # Piped input: always use env value without prompting
            print(f"  {GN}{label}{R}: {masked}  {GR}(from .env){R}")
            return name, env_val

    # Not in env — prompt (requires interactive terminal for secrets)
    if p.get("optional") and not default:
        print(f"  {GR}{label} (optional — press Enter to skip){R}")
        val = _ask("  > ", default=default, secret=secret)
        return name, val

    val = _ask(f"  {label}", default=default, secret=secret)
    if not val and not p.get("optional"):
        print(f"{RD}  Required — cannot be empty.{R}")
        return _collect_param(p)   # re-ask
    return name, val


# ── Provider / model setup ────────────────────────────────────────────────────

def setup_llm(args: argparse.Namespace):
    """Interactive provider + model selection; returns a BaseChatModel."""
    from langchain.chat_models import init_chat_model

    # ── 1. Provider selection ──────────────────────────────────────────────
    if args.provider:
        pid = args.provider.lower().replace("-", "_")
        if pid not in PROVIDERS:
            print(f"{RD}Unknown provider {pid!r}. Valid: {', '.join(PROVIDER_NAMES)}{R}")
            sys.exit(1)
    else:
        print(f"\n{B}Select a provider:{R}")
        for i, (k, v) in enumerate(PROVIDERS.items(), 1):
            print(f"  {i:2d}.  {v['label']:35s}  {GR}({k}){R}")
        raw = _ask("\nEnter number or provider name")
        if raw.isdigit():
            idx = int(raw) - 1
            if not 0 <= idx < len(PROVIDERS):
                print(f"{RD}Invalid number.{R}"); sys.exit(1)
            pid = PROVIDER_NAMES[idx]
        else:
            pid = raw.lower().replace("-", "_")
            if pid not in PROVIDERS:
                print(f"{RD}Unknown provider.{R}"); sys.exit(1)

    cfg = PROVIDERS[pid]
    print(f"\n{B}{cfg['label']}{R}")

    # ── 2. Model name ──────────────────────────────────────────────────────
    if args.model:
        model_name = args.model
    else:
        prompt_label = "Deployment name" if pid == "azure_openai" else "Model name"
        model_name = _ask(f"  {prompt_label}", default=cfg["default_model"])
        if not model_name:
            print(f"{RD}Model name is required.{R}"); sys.exit(1)

    # ── 3. Required params ─────────────────────────────────────────────────
    kwargs: dict[str, Any] = {}
    for p in cfg["params"]:
        name, val = _collect_param(p)
        if val:
            kwargs[name] = val

    # ── 4. Optional generation params ─────────────────────────────────────
    print(f"\n  {GR}Generation params (Enter to keep defaults):{R}")
    for param_name, default_val in cfg["optional"].items():
        raw = _ask(f"    {param_name}", default=str(default_val))
        try:
            kwargs[param_name] = type(default_val)(raw)
        except (ValueError, TypeError):
            pass   # keep default already set

    # ── 5. Build model ─────────────────────────────────────────────────────
    model_spec = f"{pid}:{model_name}"
    try:
        llm = init_chat_model(model_spec, **kwargs)
        print(f"\n  {GN}Model ready:{R} {model_spec}\n")
        return llm, model_spec
    except ImportError as e:
        pkg = str(e).split("'")[1] if "'" in str(e) else "required package"
        print(f"{RD}Missing package: {pkg}\n  Run: uv add {pkg.replace('.', '-')}{R}")
        sys.exit(1)
    except Exception as e:
        print(f"{RD}Failed to initialise model: {e}{R}")
        sys.exit(1)


# ── Agent setup ───────────────────────────────────────────────────────────────

def setup_agent(llm, skills_dir: Path | None):
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver
    from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

    root = skills_dir or Path.cwd()
    print(f"  {GR}Skills root: {root}{R}")
    print(f"  {GR}Scanning for SKILL.md files...{R}", end="", flush=True)

    backend = SessionLocalShellBackend(root_dir=root)

    # Peek at what skills exist before creating agent
    rmd = RecursiveSkillsMiddleware(backend=backend, sources=["/"])
    class _FakeRuntime:
        context = None
        stream_writer = None
        store = None
    skill_list = rmd._collect_all_skills(backend, "/")
    if skill_list:
        print(f" {GN}found {len(skill_list)} skill(s): {', '.join(s['name'] for s in skill_list)}{R}")
    else:
        print(f" {GR}none found (agent will still work without skills){R}")

    # create_deep_agent emits debug prints from inside deepagents — suppress them
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        agent = create_deep_agent(
            model=llm,
            backend=backend,
            skills=None,
            middleware=[rmd],
            checkpointer=MemorySaver(),
        )
    return agent, backend, skill_list


# ── Streaming renderer ────────────────────────────────────────────────────────

def stream_response(agent, messages: list, thread_id: str):
    """Stream one agent turn; return the final assistant text."""
    # Track pending tool calls: id -> name (accumulated from chunks)
    pending_tools: dict[str, str] = {}
    printed_tools: set[str] = set()
    text_buf: list[str] = []
    last_was_text = False

    try:
        for chunk, meta in agent.stream(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            cls = chunk.__class__.__name__

            # ── AI text tokens ─────────────────────────────────────────────
            if cls == "AIMessageChunk":
                content = chunk.content
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")

                if text:
                    if not last_was_text:
                        # First text token after a tool result — add separator
                        print(f"\n{CY}{B}Assistant:{R} ", end="", flush=True)
                    print(text, end="", flush=True)
                    text_buf.append(text)
                    last_was_text = True

                # ── Tool call chunks ───────────────────────────────────────
                for tcc in getattr(chunk, "tool_call_chunks", []):
                    tc_id  = tcc.get("id") or str(tcc.get("index", ""))
                    tc_name = tcc.get("name", "")
                    if tc_name:
                        pending_tools[tc_id] = tc_name
                    # Show each unique tool call once (when name first arrives)
                    tool_key = tc_id + (tc_name or "")
                    if tc_name and tool_key not in printed_tools:
                        printed_tools.add(tool_key)
                        if last_was_text:
                            print()   # end the streaming text line
                            last_was_text = False
                        # Collect args from subsequent chunks (best-effort)
                        args_preview = str(tcc.get("args", ""))[:80]
                        print(f"\n  {YL}  calling {tc_name}({args_preview}{'...' if len(str(tcc.get('args',''))) > 80 else ''}){R}",
                              flush=True)

            # ── Tool results (complete ToolMessages) ───────────────────────
            elif cls == "ToolMessage":
                tool_name = getattr(chunk, "name", "tool")
                raw = str(chunk.content)
                # Show first non-empty line up to 120 chars
                first_line = next((l for l in raw.splitlines() if l.strip()), raw)[:120]
                status = "ok" if not first_line.startswith("Error") else "ERROR"
                color  = GR if status == "ok" else RD
                print(f"\n  {color}  {tool_name} [{status}]: {first_line}{R}", flush=True)
                last_was_text = False

    except KeyboardInterrupt:
        print(f"\n{GR}(interrupted){R}")

    if last_was_text:
        print()   # final newline after streamed text

    return "".join(text_buf)


# ── Chat loop ─────────────────────────────────────────────────────────────────

def chat_loop(agent, skill_list: list):
    import uuid
    thread_id = str(uuid.uuid4())
    history: list[dict] = []

    print(f"\n{B}Chat started.{R}  {GR}Commands: /quit  /clear  /skills  /help{R}\n")

    # Try to enable readline history (Linux/macOS)
    try:
        import readline   # noqa: F401
    except ImportError:
        pass

    while True:
        try:
            user_input = input(f"{GN}You:{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GR}Goodbye.{R}")
            break

        if not user_input:
            continue

        # ── Commands ───────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print(f"{GR}Goodbye.{R}")
            break

        if user_input.lower() == "/clear":
            history.clear()
            thread_id = str(uuid.uuid4())
            print(f"{GR}Conversation cleared.{R}\n")
            continue

        if user_input.lower() == "/skills":
            if skill_list:
                for s in skill_list:
                    print(f"  {YL}{s['name']}{R}: {s['description'][:80]}")
            else:
                print(f"  {GR}No skills loaded.{R}")
            print()
            continue

        if user_input.lower() == "/help":
            print(f"  {GN}/quit /exit{R}  — exit")
            print(f"  {GN}/clear{R}       — reset conversation")
            print(f"  {GN}/skills{R}      — list loaded skills")
            print(f"  {GN}/help{R}        — this message\n")
            continue

        # ── Normal message ─────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        final_text = stream_response(agent, list(history), thread_id)
        if final_text:
            history.append({"role": "assistant", "content": final_text})
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLI chatbot powered by deepagents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported providers: " + ", ".join(PROVIDER_NAMES),
    )
    parser.add_argument("--provider", "-p", help="Provider name (e.g. google_genai, openai, azure_openai, ollama)")
    parser.add_argument("--model",    "-m", help="Model name or Azure deployment name")
    parser.add_argument("--skills",   "-s", help="Project directory to load skills from (default: cwd)")
    parser.add_argument("--no-skills",      action="store_true", help="Disable skills discovery")
    args = parser.parse_args()

    print(f"\n{B}Deep Agent CLI{R}  {GR}(deepagents + LangGraph){R}")
    print("-" * 50)

    llm, model_spec = setup_llm(args)

    skills_dir: Path | None = None
    if not args.no_skills:
        if args.skills:
            skills_dir = Path(args.skills).resolve()
        else:
            skills_dir = Path.cwd()

    agent, backend, skill_list = setup_agent(llm, skills_dir)
    chat_loop(agent, skill_list)


if __name__ == "__main__":
    main()
