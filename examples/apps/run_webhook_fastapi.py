# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ipdb",
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "flyte==2.0.0b56",
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
    # Startup: Initialize Flyte with passthrough authentication
    await flyte.init_passthrough.aio(
        project=flyte.current_project(),
        domain=flyte.current_domain(),
    )
    logger.info("Initialized Flyte passthrough auth")
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
    import httpx

    flyte.init_from_config(log_level=logging.DEBUG)

    flyte.deploy(task_env)

    # deploy the environments
    served_app = flyte.serve(app_env)
    url = served_app.url
    endpoint = served_app.endpoint
    print(f"Webhook is served on {endpoint}. you can check logs, status etc {endpoint}")

    # Use a Flyte user token for passthrough auth (instead of static API key)
    token = os.getenv("FLYTE_API_KEY")
    if not token:
        raise ValueError("FLYTE_API_KEY not set. Obtain with: flyte get api-key")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "flyte-webhook-client/1.0",
    }

    with httpx.Client(headers=headers) as client:
        # Test /me endpoint to verify passthrough auth works
        me_resp = client.get(endpoint.rstrip("/") + "/me")
        me_resp.raise_for_status()
        print(f"/me response: {me_resp.text}")

        # Test /run-task endpoint
        data = {"x": 42, "y": "hello"}
        route = "/run-task/flytesnacks/development/webhook-runner-task.webhook_task"
        full_url = endpoint.rstrip("/") + route
        print(full_url)

        resp = client.post(full_url, json=data)
        if resp.is_success:
            print(f"Webhook response: {resp.text}")
        else:
            print(f"HTTP Error: {resp.status_code} - {resp.text}")
