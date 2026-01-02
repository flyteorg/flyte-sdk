# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ipdb",
#     "fastapi",
#     "uvicorn",
#     "flyte==2.0.0b42",
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
import flyte.errors
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment

WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY", "test-api-key")
security = HTTPBearer()
logger = logging.getLogger(__name__)


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(security)],
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
    await flyte.init_in_cluster.aio(org=os.environ.get("ORG", "demo"))
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


@app.post("/run-task/{project}/{domain}/{name}")
async def run_task(
    project: str,
    domain: str,
    name: str,
    inputs: dict,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
    version: str | None = None,
):
    """
    Trigger a Flyte task run via webhook.

    This endpoint launches a Flyte task and returns information about the launched run,
    including the URL to view the run in the Flyte UI and the run's unique ID.

    Args:
        project: Flyte project name
        domain: Flyte domain (e.g., development, staging, production)
        name: Task name
        inputs: Dictionary of input parameters for the task
        credentials: Bearer token for authentication
        version: Task version

    Returns:
        Dictionary containing the launched run information:
        - url: URL to view the run in the Flyte UI
        - name: Name of the run
    """
    logger.info(f"Running task: {name} {version}, with inputs: {inputs}")
    try:
        if version is None:
            auto_version = "latest"
        else:
            auto_version = None
        tk = remote.Task.get(project=project, domain=domain, name=name, version=version, auto_version=auto_version)
        r = await flyte.run.aio(tk, **inputs)
    except flyte.errors.RemoteTaskError:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return {"url": r.url, "name": r.name}


image = flyte.Image.from_uv_script(__file__, name="webhook-runner", pre=True)

task_env = flyte.TaskEnvironment(
    name="webhook-runner-task",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

app_env = FastAPIAppEnvironment(
    name="webhook-runner",
    app=app,
    description="A webhook service that triggers Flyte task runs",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    env_vars={
        "WEBHOOK_API_KEY": os.getenv("WEBHOOK_API_KEY", "test-api-key"),
        "ORG": os.environ.get("ORG", "demo"),
    },
    depends_on=[task_env],
)


@task_env.task
async def webhook_task(x: int, y: str) -> str:
    """Run a Flyte task via webhook."""
    return f"{x!s} {y}"


if __name__ == "__main__":
    import json
    import time
    import urllib.error
    import urllib.request

    import flyte.remote

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent, log_level=logging.DEBUG)

    # deploy the environments
    deployment_list = flyte.deploy(app_env)
    d = deployment_list[0]
    app_deployment = deployment_list[0].envs["webhook-runner"]
    print(f"Deployed Webhook Runner app: {app_deployment.table_repr()}")
    url = app_deployment.deployed_app.endpoint

    # wait for the app to be active
    while True:
        app = flyte.remote.App.get(project="flytesnacks", domain="development", name="webhook-runner")
        if app.is_active():
            print("App is active")
            break
        time.sleep(1)

    data = {"x": 42, "y": "hello"}
    # Use the same token that was set when the app was deployed
    token = os.getenv("WEBHOOK_API_KEY", "test-api-key")
    route = "/run-task/flytesnacks/development/webhook-runner-task.webhook_task"
    full_url = url.rstrip("/") + route
    print(full_url)

    req = urllib.request.Request(
        full_url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "flyte-webhook-client/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"Webhook response: {resp.read().decode('utf-8')}")
    except urllib.error.HTTPError as http_err:
        print(f"HTTP Error: {http_err.code} - {http_err.read().decode('utf-8')}")
