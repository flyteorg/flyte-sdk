import logging
import os
import pathlib
import typing

import httpx
from fastapi import FastAPI

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx")

app1 = FastAPI(
    title="App 1",
    description="A FastAPI app that runs some computations",
)

app2 = FastAPI(
    title="App 2",
    description="A FastAPI app that proxies requests to another FastAPI app",
)

env1 = FastAPIAppEnvironment(
    name="app1-is-called-by-app2",
    app=app1,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)

env2 = FastAPIAppEnvironment(
    name="app2-calls-app1",
    app=app2,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    parameters=[
        flyte.app.Parameter(
            name="app1_url",
            value=flyte.app.AppEndpoint(app_name="app1-is-called-by-app2"),
            env_var="APP1_URL",
        ),
    ],
    depends_on=[env1],
    env_vars={"LOG_LEVEL": "10"},
)


@app1.get("/greeting/{name}")
async def greeting(name: str) -> str:
    return f"Hello, {name}!"


@app2.get("/app1-endpoint")
async def get_app1_endpoint() -> str:
    return env1.endpoint


@app2.get("/greeting/{name}")
async def greeting_proxy(name: str) -> typing.Any:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{env1.endpoint}/greeting/{name}")
        return response.json()


@app2.get("/app1-url")
async def get_app1_url() -> str:
    return os.getenv("APP1_URL")


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    app = flyte.serve(env2)
    print(f"Deployed FastAPI app: {app.url}")
