"""
Agent Handoff Web UI

A FastAPI app that provides a web interface for the agent handoff system.
Users can select from example queries or enter custom queries, trigger the
workflow, and view the results in the Flyte UI.
"""

import pathlib

# Import agent registry from main module
import sys
from contextlib import asynccontextmanager
from typing import List

import flyte
import flyte.remote as remote
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agent_handoff import AGENT_REGISTRY, EXAMPLE_QUERIES

run_handoff_wf = remote.Task.get("agent-handoff.run_handoff", version="4ffb7037519238cf7e6dafc943939e65")


class QueryRequest(BaseModel):
    """Request model for running agent handoff."""

    query: str
    threshold: float = 0.6


class AgentInfo(BaseModel):
    """Agent information for frontend display."""

    id: str
    name: str
    description: str
    tags: List[str]


class RunStatus(BaseModel):
    """Status of a Flyte run."""

    run_id: str
    url: str
    status: str
    result: dict | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte in cluster before accepting requests."""
    # Startup: Initialize Flyte
    await flyte.init_in_cluster.aio(org="demo", project="ketan", domain="development")
    await run_handoff_wf.fetch.aio()
    yield
    # Shutdown: Clean up if needed


app = FastAPI(
    title="Agent Handoff UI",
    description="Web interface for intelligent agent routing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML interface."""
    html_path = pathlib.Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/agents")
async def get_agents() -> List[AgentInfo]:
    """Get list of all available agents."""
    return [
        AgentInfo(id=agent.id, name=agent.name, description=agent.description, tags=agent.tags)
        for agent in AGENT_REGISTRY
    ]


@app.get("/api/examples")
async def get_examples() -> List[str]:
    """Get list of example queries."""
    return EXAMPLE_QUERIES


@app.post("/api/run")
async def run_handoff(request: QueryRequest) -> RunStatus:
    """
    Trigger agent handoff workflow and return run information.

    Args:
        request: Query and threshold parameters

    Returns:
        Run status with URL for viewing in Flyte UI
    """
    try:
        # Trigger the workflow asynchronously
        run = await flyte.run.aio(run_handoff_wf, query=request.query, threshold=request.threshold)

        return RunStatus(
            run_id=run.name,
            url=run.url,
            status="RUNNING",
            result={"message": "Workflow started successfully. Click the link below to view progress in Flyte UI."},
            error=None,
        )
    except Exception as e:
        print(f"Got exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


current_file = pathlib.Path(__file__)
# Configure FastAPI app environment
env = FastAPIAppEnvironment(
    name="agent-handoff-ui",
    app=app,
    description="Web UI for agent handoff system",
    image=flyte.Image.from_debian_base()
    .with_uv_project(pyproject_file=pathlib.Path(__file__).parent / "pyproject.toml", pre=True)
    .with_local_v2(),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    include=[str(current_file.parent / "static" / "index.html"), str(current_file.parent / "agent_handoff.py")],
    secrets=flyte.Secret(key="EAGER_API_KEY", as_env_var="EAGER_API_KEY"),
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=current_file.parent)
    app = flyte.serve(env)
    print(f"🚀 Deployed Agent Handoff UI: {app.url}")
    print(f"🌐 Open in browser: {app.endpoint}")
