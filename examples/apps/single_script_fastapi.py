import logging
import pathlib
from typing import Dict

from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="UV Script FastAPI Demo", description="A simple FastAPI app using UV inline script metadata", version="1.0.0"
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Hello from UV Script FastAPI!", "info": "This app is powered by UV inline script dependencies"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None) -> Dict:
    """Example endpoint with path and query parameters."""
    result = {"item_id": item_id}
    if q:
        result["q"] = q
    return result


env = FastAPIAppEnvironment(
    name="fastapi-script",
    app=app,
    description="A FastAPI app demonstrating UV inline script capabilities.",
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
    print(f"Deployed FastAPI app: {d.env_repr()}")
