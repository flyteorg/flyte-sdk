"""Local serving example where a FastAPI app calls a Flyte task.

This example demonstrates how to serve a FastAPI app locally that invokes a
Flyte task from one of its endpoints.  The task is executed via
``flyte.with_runcontext(mode="local").run(...)`` so it runs in the same process.

Usage (SDK):
    python examples/apps/local_app_calling_task.py
"""

import httpx
from fastapi import FastAPI

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="App Calling Task",
    description="A FastAPI app that delegates computation to a Flyte task",
    version="1.0.0",
)

app_env = FastAPIAppEnvironment(
    name="local-app-calling-task",
    app=app,
    description="App that calls a Flyte task to double a number",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="local-doubler-task-env",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@task_env.task
async def double(x: int) -> int:
    """A simple task that doubles the input."""
    return x * 2


@app.get("/")
async def double_endpoint(x: int) -> dict[str, int]:
    """Endpoint that invokes the ``double`` task and returns the result."""
    result = flyte.with_runcontext(mode=flyte.app.ctx().mode).run(double, x=x)
    return {"result": result.outputs()[0]}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Serve the app locally (non-blocking)
    local_app = flyte.with_servecontext(mode="local").serve(app_env)

    # Wait for the app to be ready
    local_app.activate(wait=True)
    print(f"App is ready at {local_app.endpoint}")

    # Call the app endpoint which internally runs the task
    response = httpx.get(f"{local_app.endpoint}", params={"x": 21})
    response.raise_for_status()
    data = response.json()
    print(f"Response: {data}")
    assert data["result"] == 42

    # Shut down the local app
    local_app.deactivate(wait=True)
    print("Done!")
