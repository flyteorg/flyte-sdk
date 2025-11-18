import logging
import pathlib
import typing

import httpx
from fastapi import FastAPI

import flyte
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
    requires_auth=False,
)

env2 = FastAPIAppEnvironment(
    name="app2-calls-app1",
    app=app2,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    depends_on=[env1],
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


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    deployments = flyte.deploy(env2)
    d = deployments[0]
    print(f"Deployed FastAPI app: {d.env_repr()}")
