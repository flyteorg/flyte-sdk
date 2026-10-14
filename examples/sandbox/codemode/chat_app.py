"""Chat Analytics Agent — Code Mode as a flyte.app
===================================================

A persistent chat UI served as a ``flyte.app`` (FastAPI) where users
interactively ask data-analysis questions and get back Chart.js
visualizations, HTML tables and text summaries.

The core pattern is identical to ``llm_code_mode.py``: single LLM call
-> Python code string -> Monty sandbox execution with tool functions
-> charts + summary.  The difference is that this example wraps it in a
conversational web interface that streams progress for each phase.

Architecture::

    Browser (Chat UI)
      |
      +-- GET /              -> Embedded HTML/CSS/JS chat interface
      +-- GET /api/config    -> {"model": ...} shown in the header
      +-- GET /api/tools     -> JSON list of available tool descriptions
      +-- GET /api/datasets  -> JSON list of demo datasets (sidebar)
      +-- POST /api/chat     -> Server-Sent Events, one per phase:
             |                  llm_start / llm_done / exec_start / retry / done
             +-- CodeModeAgent.run_streaming(message, history)
                    +-- LLM call (generate code)
                    +-- flyte.sandbox.orchestrate_local(code, tasks=ALL_TOOLS)
                    +-- retry on failure (up to max_retries)

Install dependencies::

    pip install 'flyte[sandbox]' fastapi uvicorn httpx

Run locally (needs ANTHROPIC_API_KEY in the environment)::

    python examples/sandbox/codemode/chat_app.py

Deploy as a Flyte app::

    python examples/sandbox/codemode/chat_app.py deploy
"""

import json
import pathlib
import sys

from _agent import CodeModeAgent
from _tools import ALL_TOOLS, dataset_catalog
from _ui import CHAT_HTML
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import flyte
import flyte.app
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
        "flyte[sandbox]",
    ),
    secrets=flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    scaling=flyte.app.Scaling(replicas=1),
)

agent = CodeModeAgent(tools=ALL_TOOLS, max_retries=2)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/config")
async def get_config() -> dict[str, str]:
    """Runtime configuration surfaced in the UI header."""
    return {"model": agent.model, "sandbox": "Monty (flyte.sandbox)"}


@app.get("/api/tools")
async def get_tools() -> list[dict]:
    """Return JSON descriptions of available tool functions."""
    return agent.tool_descriptions()


@app.get("/api/datasets")
async def get_datasets() -> list[dict]:
    """Return the demo datasets with their columns and row counts."""
    return dataset_catalog()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Core endpoint: generate code, run in sandbox, stream phase events as SSE."""

    async def event_stream():
        async for event in agent.run_streaming(req.message, req.history):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the embedded chat UI."""
    return HTMLResponse(content=CHAT_HTML)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
        app_handle = flyte.serve(env)
        print(f"Deployed Chat Analytics Agent: {app_handle.url}")
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
