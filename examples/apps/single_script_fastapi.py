# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "flyte>=2.0.0b29"
# ]
# ///

import logging
import pathlib

from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="Single script FastAPI Demo", description="A simple FastAPI app using a single script", version="1.0.0"
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Hello from Single-script FastAPI!", "info": "This app is powered by a single script"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None) -> dict:
    """Example endpoint with path and query parameters."""
    result = {"item_id": item_id}
    if q:
        result["q"] = q
    return result


env = FastAPIAppEnvironment(
    name="fastapi-script",
    app=app,
    description="A FastAPI app demonstrating UV inline script capabilities.",
    # image=flyte.Image.from_uv_script(__file__, name="fastapi-script"),
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    deployments = flyte.deploy(env)
    d = deployments[0]
    print(f"Deployed FastAPI app: {d.table_repr()}")
