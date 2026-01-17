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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette import status

import flyte
import flyte.errors
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment, FastAPIPassthroughAuthMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager to initialize Flyte with passthrough auth.

    This initializes Flyte with passthrough authentication, allowing the app to
    pass user credentials from incoming requests to the Flyte control plane.
    """
    ENDPOINT_OVERRIDE_ENV_VAR = "_U_EP_OVERRIDE"
    PROJECT_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_PROJECT"
    DOMAIN_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_DOMAIN"

    # Startup: Initialize Flyte with passthrough authentication
    endpoint = os.getenv(ENDPOINT_OVERRIDE_ENV_VAR, None)
    if not endpoint:
        raise RuntimeError(f"Endpoint could not be determined from {ENDPOINT_OVERRIDE_ENV_VAR!r} environment variable")
    await flyte.init_passthrough.aio(
        endpoint=endpoint,
        project=os.getenv(PROJECT_NAME_ENV_VAR, None),
        domain=os.getenv(DOMAIN_NAME_ENV_VAR, None),
    )
    logger.info(f"Initialized Flyte passthrough auth to {endpoint}")
    yield
    # Shutdown: Clean up if needed


app = FastAPI(
    title="Flyte Webhook Runner",
    description="A webhook service that triggers Flyte task runs",
    version="1.0.0",
    lifespan=lifespan,
)

# Add auth middleware - automatically extracts auth headers and sets Flyte context
app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/health"})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/me")
async def get_current_user():
    """
    Get information about the currently authenticated user.

    Verifies passthrough authentication by fetching user info from the
    Flyte control plane using the caller's credentials.
    """
    try:
        # Auth metadata automatically set by FastAPIAuthMiddleware
        user = await remote.User.get.aio()
        return {
            "subject": user.subject(),
            "name": user.name(),
        }
    except Exception as e:
        logger.error(f"Failed to get user info: {type(e).__name__}")
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

    This endpoint launches a Flyte task using passthrough authentication,
    meaning the task is executed with the permissions of the calling user.

    Args:
        project: Flyte project name
        domain: Flyte domain (e.g., development, staging, production)
        name: Task name
        inputs: Dictionary of input parameters for the task
        version: Task version (optional, defaults to "latest")

    Returns:
        Dictionary containing the launched run information:
        - url: URL to view the run in the Flyte UI
        - name: Name of the run
    """
    logger.info(f"Running task: {project}/{domain}/{name} version={version}")

    try:
        if version is None:
            auto_version = "latest"
        else:
            auto_version = None

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
            detail=f"Task {name} with {version} in {project} and {domain} not found",
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


# image = flyte.Image.from_uv_script(__file__, name="webhook-runner", pre=True)
image = flyte.Image.from_debian_base().with_pip_packages("ipdb", "fastapi", "uvicorn")

task_env = flyte.TaskEnvironment(
    name="webhook-runner-task",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

app_env = FastAPIAppEnvironment(
    name="webhook-runner",
    app=app,
    description="A webhook service that triggers Flyte task runs with passthrough auth",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,  # Platform handles auth at gateway
    depends_on=[task_env],
    scaling=flyte.app.Scaling(replicas=1),
)


@task_env.task
async def webhook_task(x: int, y: str) -> str:
    """Run a Flyte task via webhook."""
    return f"{x!s} {y}"


if __name__ == "__main__":
    import json
    import urllib.error
    import urllib.request

    import flyte.remote

    flyte.init_from_config(log_level=logging.DEBUG)

    # deploy the environments
    served_app = flyte.serve(app_env)
    url = served_app.url
    endpoint = served_app.endpoint
    print(f"Webhook is served on {endpoint}. you can check logs, status etc {endpoint}")

    # Use a Flyte user token for passthrough auth (instead of static API key)
    token = os.getenv("FLYTE_API_KEY")
    if not token:
        raise ValueError("FLYTE_API_KEY not set. Obtain with: flyte get api-key")

    # Test /me endpoint to verify passthrough auth works
    me_req = urllib.request.Request(
        url.rstrip("/") + "/me",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(me_req) as resp:
        print(f"/me response: {resp.read().decode('utf-8')}")

    # Test /run-task endpoint
    data = {"x": 42, "y": "hello"}
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
