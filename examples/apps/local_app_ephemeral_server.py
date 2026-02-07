"""Local serving example using ephemeral_server context manager.

This example demonstrates how to serve a FastAPI app locally using the
ephemeral_server() context manager, which ensures the app is properly
activated and deactivated. The FastAPI lifespan simulates loading a model
on startup.

Usage (SDK):
    python examples/apps/local_app_ephemeral_server.py

Usage (CLI):
    flyte serve --local examples/apps/local_app_ephemeral_server.py app_env
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan function that simulates loading a model on startup."""
    print("Simulate loading model...")
    await asyncio.sleep(3)
    app.state.model = {"m": 5, "b": 11}
    print("Model loaded!")
    yield
    print("Shutting down, releasing model resources...")


app = FastAPI(
    title="Local Linear Regression",
    description="A local FastAPI app that performs linear regression",
    version="1.0.0",
    lifespan=lifespan,
)

app_env = FastAPIAppEnvironment(
    name="local-linear-regression",
    app=app,
    description="Performs linear regression",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="local-linear-regression-task-env",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@app.post("/predict")
async def predict(x: int) -> dict[str, int]:
    """Perform linear regression."""
    result = app.state.model["m"] * x + app.state.model["b"]
    return {"result": result}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@task_env.task
async def predict_task(data: list[int]) -> list[int]:
    """Task that calls the local app endpoint."""
    print(f"Calling app at {app_env.endpoint}")
    async with httpx.AsyncClient() as client:
        results = []
        for x in data:
            response = await client.post(f"{app_env.endpoint}/predict", params={"x": x})
            response.raise_for_status()
            results.append(response.json()["result"])
        return results


if __name__ == "__main__":
    # Serve the app locally (non-blocking)
    local_app = flyte.with_servecontext(mode="local").serve(app_env)

    # Use ephemeral_server to ensure the app is activated and deactivated
    async def main():
        async with local_app.ephemeral_context():
            print(f"App is ready at {local_app.endpoint}")

            # Call the app endpoint directly
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{local_app.endpoint}/predict", params={"x": 5})
                response.raise_for_status()
                print(f"Direct call result: {response.json()}")

            # Run a task that calls the local app
            result = await flyte.with_runcontext(mode="local").run.aio(predict_task, data=[5, 10, 15])
            print(f"Task result: {result.outputs()[0]}")
            assert result.outputs()[0] == [36, 61, 86]

        assert local_app.is_deactivated()

    asyncio.run(main())
    # App is automatically deactivated after exiting the context
    print("Done!")
