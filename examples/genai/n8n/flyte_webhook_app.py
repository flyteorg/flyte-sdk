"""
Flyte Webhook App for n8n Integration

A FastAPI-based webhook service that allows n8n workflows to trigger Flyte task
runs via HTTP. Uses passthrough authentication so that the caller's credentials
are forwarded to the Flyte control plane when launching tasks.

Usage:
    # Deploy
    python flyte_webhook_app.py

    # n8n HTTP Request node configuration:
    # - Method: POST
    # - URL: https://<subdomain>.apps.<endpoint>/run-task/{project}/{domain}/{task_name}
    # - Headers: Authorization: Bearer <token>, Content-Type: application/json
    # - Body (JSON): {"input_key": "input_value", ...}
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette import status

import flyte
import flyte.app
import flyte.errors
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment, FastAPIPassthroughAuthMiddleware

logger = logging.getLogger(__name__)

image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")


# ---------------------------------------------------------------------------
# FastAPI lifespan: initialize Flyte passthrough auth on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECT_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_PROJECT"
    DOMAIN_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_DOMAIN"

    await flyte.init_passthrough.aio(
        project=os.getenv(PROJECT_NAME_ENV_VAR, None),
        domain=os.getenv(DOMAIN_NAME_ENV_VAR, None),
    )
    logger.info("Initialized Flyte passthrough auth")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Flyte n8n Webhook Runner",
    description="A webhook service that lets n8n trigger Flyte task runs",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware: extract Authorization header and set Flyte auth context per-request
app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/health"})


@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy"}


@app.get("/me")
async def get_current_user():
    """Verify passthrough auth by fetching the current user from the Flyte control plane."""
    try:
        user = await remote.User.get.aio()
        return {
            "name": user.name,
            "subject": user.subject,
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or unauthorized",
        )


@app.post("/run-task/{project}/{domain}/{name}")
async def run_task(
    project: str,
    domain: str,
    name: str,
    inputs: dict,
    version: str | None = None,
):
    """
    Trigger a Flyte task run with the caller's credentials.

    The task is executed with the permissions of the calling user (passthrough auth).

    Path parameters:
        project: Flyte project name
        domain:  Flyte domain (e.g. development, staging, production)
        name:    Fully-qualified task name (e.g. "my_env.my_task")

    Query parameters:
        version: Task version (optional — defaults to "latest")

    Body (JSON):
        inputs:  Dictionary of input parameters for the task
    """
    logger.info(f"Running task: {project}/{domain}/{name} version={version}")
    try:
        auto_version = "latest" if version is None else None
        tk = remote.Task.get(
            project=project,
            domain=domain,
            name=name,
            version=version,
            auto_version=auto_version,
        )
        r = await flyte.run.aio(tk, **inputs)
        return {"url": r.url, "name": r.name}

    except flyte.errors.RemoteTaskNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {name} v{version} in {project}/{domain} not found",
        )
    except flyte.errors.RemoteTaskUsageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ---------------------------------------------------------------------------
# Task environment: an example task that can be triggered via the webhook
# ---------------------------------------------------------------------------
task_env = flyte.TaskEnvironment(
    name="flyte-n8n-webhook-task",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@task_env.task
async def webhook_task(x: int, y: str) -> dict:
    """Example Flyte task callable via the webhook."""
    return {"result": f"{x!s} {y}"}


@task_env.task
async def add_field(data: dict) -> dict:
    """Example Flyte task callable via the webhook."""
    data["new_flyte_field"] = "hello from flyte"
    return data


# ---------------------------------------------------------------------------
# App environment: the webhook FastAPI service
# ---------------------------------------------------------------------------
flyte_n8n_webhook_app = FastAPIAppEnvironment(
    name="flyte-n8n-webhook-app",
    app=app,
    description="A webhook service that lets n8n trigger Flyte task runs with passthrough auth",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=True,  # Platform handles auth at the gateway
    depends_on=[task_env],
    scaling=flyte.app.Scaling(replicas=(0, 1)),
    port=8080,
)


# ---------------------------------------------------------------------------
# Deploy helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import httpx
    import flyte.remote

    flyte.init_from_config()
    flyte.deploy(flyte_n8n_webhook_app)

    app = flyte.remote.App.get(name="flyte-n8n-webhook-app")
    url = app.url
    endpoint = app.endpoint
    print(f"Deployed webhook app: {url}")
    print(f"Webhook is served on {endpoint}. you can check logs, status etc {endpoint}")

    # --- Quick smoke test ---
    token = os.getenv("FLYTE_API_KEY")

    headers = {"Authorization": f"Bearer {token}"}

    # Test /health
    health_endpoint = endpoint.rstrip("/") + "/health"
    resp = httpx.get(health_endpoint, headers=headers)
    print(f"/health response: {resp.text}")

    # Test /me
    me_endpoint = endpoint.rstrip("/") + "/me"
    resp = httpx.get(me_endpoint, headers=headers)
    print(f"/me response: {resp.text}")

    # Test /run-task (triggers the example add_field)
    data = {"data": {"x": 42, "y": "hello from n8n"}}
    route = "/run-task/flytesnacks/development/flyte-n8n-webhook-task.add_field"
    full_endpoint = endpoint.rstrip("/") + route
    print(f"POST {full_endpoint}")

    resp = httpx.post(
        full_endpoint,
        json=data,
        headers=headers,
    )
    if resp.is_success:
        print(f"Webhook response: {resp.text}")
    else:
        print(f"HTTP Error: {resp.status_code} - {resp.text}")
