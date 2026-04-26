"""
Azure OpenAI — using a deployment that you created in Azure AI Foundry.

All credentials are passed directly; nothing from environment variables.
Replace every XXXX placeholder with your real Azure values.

How to find these values in the Azure portal
--------------------------------------------
azure_endpoint      : Azure AI Foundry → your resource → Keys and Endpoint → Endpoint
azure_deployment    : Azure AI Foundry → Deployments → your deployment name
api_key             : Azure AI Foundry → your resource → Keys and Endpoint → KEY 1
api_version         : https://learn.microsoft.com/azure/ai-services/openai/api-version-lifecycle

Requirements
------------
    uv add langchain-openai
"""

from pathlib import Path

from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Model ─────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    # The deployment name you gave when deploying the model in Azure AI Foundry
    azure_deployment="XXXX_DEPLOYMENT_NAME_XXXX",

    # Your Azure OpenAI resource endpoint (includes trailing slash)
    azure_endpoint="https://XXXX_RESOURCE_NAME_XXXX.openai.azure.com/",

    # API key from Azure portal (Keys and Endpoint → KEY 1)
    api_key="XXXX_AZURE_API_KEY_XXXX",

    # API version — pick the latest stable from Microsoft docs
    api_version="2025-01-01-preview",

    # Model parameters
    temperature=0.7,
    max_tokens=4096,
    top_p=0.95,

    # Optional: Azure Active Directory auth instead of api_key
    # azure_ad_token="XXXX_AAD_TOKEN_XXXX",
    # azure_ad_token_provider=some_callable,
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
    config={"configurable": {"thread_id": "azure-session-001"}},
)

for msg in result["messages"]:
    if msg.__class__.__name__ == "AIMessage":
        content = msg.content
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if content:
            print(content)
