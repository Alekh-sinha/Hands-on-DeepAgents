"""
Azure OpenAI — model deployed in Azure AI Foundry.

Uses init_chat_model as the aggregator.

How Azure maps through init_chat_model
---------------------------------------
  init_chat_model(
      "azure_openai:my-deployment",     <- provider:deployment_name
      azure_endpoint="https://...",
      api_key="...",
      api_version="...",
  )
  -> _parse_model: provider="azure_openai", model="my-deployment"
  -> AzureChatOpenAI(
         model="my-deployment",         <- used as azure_deployment if not set separately
         azure_endpoint="https://...",
         api_key="...",                 <- alias for openai_api_key
         api_version="...",
     )

Why the explicit "azure_openai:" prefix is required
-----------------------------------------------------
_attempt_infer_model_provider() has no rule for Azure deployment names (they are
arbitrary strings like "gpt-4o-prod" or "my-chat-model").  Auto-inference always
returns None.  The "provider:model" format is the only safe approach for Azure.

How to find your values
-----------------------
  azure_endpoint    : Azure AI Foundry → resource → Keys and Endpoint → Endpoint
  deployment name   : Azure AI Foundry → Deployments → your deployment → name
  api_key           : Azure AI Foundry → resource → Keys and Endpoint → KEY 1
  api_version       : https://learn.microsoft.com/azure/ai-services/openai/api-version-lifecycle

Requirements
------------
  uv add langchain-openai
"""

from pathlib import Path

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from session_backend import SessionLocalShellBackend, RecursiveSkillsMiddleware

# ── Model ─────────────────────────────────────────────────────────────────────
llm = init_chat_model(
    # "azure_openai:<deployment-name>" — the part after the colon is your deployment
    "azure_openai:XXXX_DEPLOYMENT_NAME_XXXX",

    # Azure resource endpoint
    azure_endpoint="https://XXXX_RESOURCE_NAME_XXXX.openai.azure.com/",

    # API key from Azure portal (Keys and Endpoint → KEY 1 or KEY 2)
    api_key="XXXX_AZURE_API_KEY_XXXX",

    # REST API version — use the latest stable from Microsoft docs
    api_version="2025-01-01-preview",

    # Generation parameters — same kwargs as every other provider
    temperature=0.7,
    max_tokens=4096,
    top_p=0.95,

    # Optional: Azure Active Directory token auth instead of api_key
    # azure_ad_token="XXXX_AAD_TOKEN_XXXX",

    # Optional: use a different deployment for streaming vs non-streaming
    # streaming=True,
)

# ── Backend ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/path/to/your/project")   # ← replace

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
    config={"configurable": {"thread_id": "azure-session-001"}},
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
