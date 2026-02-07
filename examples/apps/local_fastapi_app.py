"""Local serving example using FastAPIAppEnvironment.

This example demonstrates how to serve a FastAPI app locally and call its
endpoint from a Flyte task, all within the same script.

Usage (SDK):
    python examples/apps/local_fastapi_app.py

Usage (CLI):
    flyte serve --local examples/apps/local_fastapi_app.py app_env
"""

import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="Local Add One",
    description="A local FastAPI app that adds one to the input",
    version="1.0.0",
)

app_env = FastAPIAppEnvironment(
    name="local-add-one",
    app=app,
    description="Adds one to the input",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="local-add-one-task-env",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@app.get("/")
async def add_one(x: int) -> dict[str, int]:
    """Add one to the input."""
    return {"result": x + 1}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@task_env.task
async def add_one_task(x: int) -> int:
    """Task that calls the local app endpoint."""
    print(f"Calling app at {app_env.endpoint}")
    async with httpx.AsyncClient() as client:
        response = await client.get(app_env.endpoint, params={"x": x})
        response.raise_for_status()
        return response.json()["result"]


if __name__ == "__main__":
    # Serve the app locally (non-blocking)
    local_app = flyte.with_servecontext(mode="local").serve(app_env)

    # Wait for the app to be ready
    local_app.activate()
    print(f"App is ready at {local_app.endpoint}")

    # Run a task that calls the local app
    result = flyte.with_runcontext(mode="local").run(add_one_task, x=5)
    print(f"Result: {result.outputs()[0]}")
    assert result.outputs()[0] == 6

    # Shut down the local app
    local_app.deactivate()
    print("Done!")
