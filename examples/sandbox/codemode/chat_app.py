"""Chat Analytics Agent — Code Mode as a flyte.app
===================================================

A persistent chat UI served as a ``flyte.app`` (FastAPI) where users
interactively ask data-analysis questions and get back Chart.js
visualizations + text summaries.

The core pattern is identical to ``llm_code_mode.py``: single LLM call
-> Python code string -> Monty sandbox execution with tool functions
-> charts + summary.  The difference is that this example wraps it in a
conversational web interface instead of a one-shot ``flyte.run``.

Architecture::

    Browser (Chat UI)
      |
      +-- GET /           -> Embedded HTML/CSS/JS chat interface
      +-- GET /api/tools  -> JSON list of available tool descriptions
      +-- POST /api/chat  -> { message, history } -> { code, charts, summary, error }
             |
             +-- CodeModeAgent.run(message, history)
                    +-- LLM call (generate code)
                    +-- run_local_sandbox(code, functions=ALL_TOOLS)
                    +-- retry on failure (up to max_retries)

Install dependencies::

    pip install 'flyte[sandbox]' anthropic

Run::

    python examples/sandbox/codemode/chat_app.py
"""

import pathlib

from _agent import CodeModeAgent
from _tools import ALL_TOOLS
from _ui import CHAT_HTML
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# ---------------------------------------------------------------------------
# FastAPI + AppEnvironment setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Chat Data Analytics Agent")

env = FastAPIAppEnvironment(
    name="chat-analytics-agent",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic-monty",
    ),
    secrets=flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    scaling=flyte.app.Scaling(replicas=1),
)

agent = CodeModeAgent(tools=ALL_TOOLS, max_retries=2)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    code: str = ""
    charts: list[str] = []
    summary: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/tools")
async def get_tools() -> list[dict]:
    """Return JSON descriptions of available tool functions."""
    return agent.tool_descriptions()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Core endpoint: generate code, run in sandbox, return results."""
    result = await agent.run(req.message, req.history)
    return ChatResponse(
        code=result.code,
        charts=result.charts,
        summary=result.summary,
        error=result.error,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the embedded chat UI."""
    return HTMLResponse(content=CHAT_HTML)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env)
    d = deployments[0]
    print(f"Deployed Chat Analytics Agent: {d.table_repr()}")
