# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "flyte>=2.0.0b27"
# ]
# ///
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status

import flyte
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment

WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY", "test-api-key")
security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    """Verify the API key from the bearer token."""
    if credentials.credentials != WEBHOOK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager to initialize Flyte before accepting requests.

    This ensures that the Flyte client is properly initialized before any requests
    are processed, preventing race conditions and initialization errors.
    """
    # Startup: Initialize Flyte
    await flyte.init_in_cluster.aio()
    yield
    # Shutdown: Clean up if needed


app = FastAPI(
    title="Flyte Webhook Runner",
    description="A webhook service that triggers Flyte task runs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/run-task/{project}/{domain}/{name}/{version}")
async def run_task(
    project: str,
    domain: str,
    name: str,
    version: str,
    inputs: dict,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
):
    """
    Trigger a Flyte task run via webhook.

    This endpoint launches a Flyte task and returns information about the launched run,
    including the URL to view the run in the Flyte UI and the run's unique ID.

    Args:
        project: Flyte project name
        domain: Flyte domain (e.g., development, staging, production)
        name: Task name
        version: Task version
        inputs: Dictionary of input parameters for the task
        credentials: Bearer token for authentication

    Returns:
        Dictionary containing the launched run information:
        - url: URL to view the run in the Flyte UI
        - id: Unique identifier for the launched run
    """
    tk = await remote.TaskDetails.fetch(project=project, domain=domain, name=name, version=version)
    r = await flyte.run.aio(tk, **inputs)
    return {"url": r.url, "id": r.id}


env = FastAPIAppEnvironment(
    name="webhook-runner",
    app=app,
    description="A webhook service that triggers Flyte task runs",
    image=flyte.Image.from_uv_script(__file__, name="webhook-runner"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    env_vars={"WEBHOOK_API_KEY": os.getenv("WEBHOOK_API_KEY", "test-api-key")},
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent, log_level=logging.DEBUG)
    deployment_list = flyte.deploy(env)
    if deployment_list:
        d = deployment_list[0]
        print(f"Deployed Webhook Runner app: {d.env_repr()}")


# TO Test
# curl -X POST "https://nameless-morning-8ff1a.apps.demo.hosted.unionai.cloud/run-task/flytesnacks/development/t1/v1" \
#   -H "Authorization: Bearer test-api-key" \
#   -H "Content-Type: application/json" \
#   -d '{"input_key": "input_value"}'
