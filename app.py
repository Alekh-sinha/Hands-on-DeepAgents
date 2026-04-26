"""Streamlit chat UI for deepagents with per-session model credentials."""

import tempfile
import uuid
import warnings
from pathlib import Path

import streamlit as st
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware, make_session_dir

# ---------------------------------------------------------------------------
# Provider / model catalogue
# ---------------------------------------------------------------------------
# Each entry: provider_id -> {label, api_key_label (None = no key), models list}
# models list: [(model_id, display_name), ...]
PROVIDERS: dict[str, dict] = {
    "google_genai": {
        "label": "Google Gemini",
        "api_key_label": "Google API Key",
        "models": [
            ("gemini-2.5-flash",  "Gemini 2.5 Flash"),
            ("gemini-2.5-pro",    "Gemini 2.5 Pro"),
            ("gemini-2.0-flash",  "Gemini 2.0 Flash"),
            ("gemini-1.5-pro",    "Gemini 1.5 Pro"),
            ("gemini-1.5-flash",  "Gemini 1.5 Flash"),
        ],
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "api_key_label": "Anthropic API Key",
        "models": [
            ("claude-sonnet-4-6",          "Claude Sonnet 4.6"),
            ("claude-opus-4-7",            "Claude Opus 4.7"),
            ("claude-haiku-4-5-20251001",  "Claude Haiku 4.5"),
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
            ("claude-3-5-haiku-20241022",  "Claude 3.5 Haiku"),
        ],
    },
    "openai": {
        "label": "OpenAI",
        "api_key_label": "OpenAI API Key",
        "models": [
            ("gpt-4o",      "GPT-4o"),
            ("gpt-4o-mini", "GPT-4o mini"),
            ("o3-mini",     "o3-mini"),
            ("o1",          "o1"),
        ],
    },
    "groq": {
        "label": "Groq",
        "api_key_label": "Groq API Key",
        "models": [
            ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
            ("mixtral-8x7b-32768",      "Mixtral 8x7B"),
            ("llama-3.1-8b-instant",    "Llama 3.1 8B"),
        ],
    },
    "ollama": {
        "label": "Ollama (local)",
        "api_key_label": None,   # no key required
        "models": [
            ("llama3.2", "Llama 3.2"),
            ("mistral",  "Mistral"),
            ("phi3",     "Phi-3"),
        ],
    },
}

PROVIDER_IDS    = list(PROVIDERS.keys())
PROVIDER_LABELS = [PROVIDERS[p]["label"] for p in PROVIDER_IDS]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Deep Agent Chat", page_icon="Robot", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "session_backend" not in st.session_state:
    st.session_state.session_backend = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuration")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()
    st.subheader("1. Provider & Model")

    # Provider dropdown
    provider_label = st.selectbox(
        "Provider",
        options=PROVIDER_LABELS,
        index=0,
    )
    provider_id = PROVIDER_IDS[PROVIDER_LABELS.index(provider_label)]
    provider_cfg = PROVIDERS[provider_id]

    # Model dropdown — options change when provider changes
    model_ids     = [m[0] for m in provider_cfg["models"]]
    model_display = [m[1] for m in provider_cfg["models"]]
    model_label   = st.selectbox("Model", options=model_display, index=0)
    model_id      = model_ids[model_display.index(model_label)]

    # API key — hidden for providers that don't need one
    api_key = ""
    if provider_cfg["api_key_label"]:
        api_key = st.text_input(
            provider_cfg["api_key_label"],
            type="password",
            placeholder="Paste your key here",
            help="Stored only in this browser session, never written to disk.",
        )

    st.divider()
    st.subheader("2. Skills Folder")
    st.caption(
        "Upload every file from your skills folder. "
        "Sub-folders with SKILL.md are auto-discovered. "
        "Supporting scripts, reference .md files, and code are all accessible."
    )
    uploaded_files = st.file_uploader(
        "Select files",
        accept_multiple_files=True,
        help="Pick every file inside your skills folder.",
    )

    st.divider()

    # Launch / restart
    if st.button("Start / Restart Agent", type="primary", use_container_width=True):

        # Validate key
        if provider_cfg["api_key_label"] and not api_key:
            st.error(f"Please enter your {provider_cfg['api_key_label']}.")
            st.stop()

        # Build the model instance with the key injected directly —
        # no environment variables are touched.
        try:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key   # alias accepted by all LangChain providers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                llm = init_chat_model(model_id, model_provider=provider_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialise model: {exc}")
            st.stop()

        # Save uploaded files into a session directory, then point the backend
        # at that directory via root_dir (new interface — no more source_dirs).
        if uploaded_files:
            tmp = Path(tempfile.mkdtemp())
            for uf in uploaded_files:
                dest = tmp / Path(uf.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(uf.read())
            root_dir = make_session_dir(
                source_dirs=[tmp],
                session_id=st.session_state.session_id,
            )
        else:
            # No upload: create an empty session dir as workspace
            root_dir = make_session_dir(
                source_dirs=[],
                session_id=st.session_state.session_id,
            )

        backend = SessionLocalShellBackend(root_dir=root_dir)
        st.session_state.session_backend = backend

        # Build the agent — use RecursiveSkillsMiddleware so skills at any
        # nesting depth inside the uploaded folder are all discovered.
        try:
            from deepagents import create_deep_agent  # noqa: PLC0415

            checkpointer = MemorySaver()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agent = create_deep_agent(
                    model=llm,
                    backend=backend,
                    skills=None,            # disable built-in one-level discovery
                    middleware=[
                        RecursiveSkillsMiddleware(backend=backend, sources=["/"]),
                    ],
                    checkpointer=checkpointer,
                )
            st.session_state.agent = agent
            st.session_state.messages = []
            st.success(f"Agent ready — {provider_cfg['label']} / {model_label}")
            st.caption(f"Sandbox: `{root_dir}`")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to create agent: {exc}")

    # Show sandbox contents
    if st.session_state.session_backend:
        sd = st.session_state.session_backend.cwd
        files = sorted(f for f in sd.rglob("*") if f.is_file())
        if files:
            with st.expander(f"Sandbox files ({len(files)})"):
                for f in files:
                    st.text(str(f.relative_to(sd)))

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Deep Agent Chat")

if st.session_state.agent is None:
    st.info(
        "Configure your provider and API key in the sidebar, "
        "optionally upload a skills folder, then click **Start / Restart Agent**."
    )
    st.stop()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask the agent anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lc_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": lc_messages},
                    config={"configurable": {"thread_id": st.session_state.session_id}},
                )
                ai_msgs = [
                    m for m in result.get("messages", [])
                    if m.__class__.__name__ == "AIMessage"
                ]
                if ai_msgs:
                    content = ai_msgs[-1].content
                    if isinstance(content, list):
                        content = "\n".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    answer = content or "_No text response._"
                else:
                    answer = "_Agent returned no message._"
            except Exception as exc:  # noqa: BLE001
                answer = f"Error: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
