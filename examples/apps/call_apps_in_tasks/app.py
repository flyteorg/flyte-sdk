import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(title="Add One", description="Adds one to the input", version="1.0.0")

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx")

app_env = FastAPIAppEnvironment(
    name="add-one-app",
    app=app,
    description="Adds one to the input",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="add_one_task_env",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[app_env],
)


@app.get("/")
async def add_one(x: int) -> dict[str, int]:
    """Main endpoint for the add-one app."""
    return {"result": x + 1}


@task_env.task
async def add_one_task(x: int) -> int:
    print(f"Calling app at {app_env.endpoint}")
    async with httpx.AsyncClient() as client:
        response = await client.get(app_env.endpoint, params={"x": x})
        response.raise_for_status()
        return response.json()["result"]
