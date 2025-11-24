import logging
import pathlib

from fastapi import FastAPI
from module import function

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(title="Multi-file FastAPI Demo", description="A FastAPI app with multiple files", version="1.0.0")

app_env = FastAPIAppEnvironment(
    name="fastapi-multi-file",
    app=app,
    description="A FastAPI app demonstrating multi-file deployments.",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)


@app.get("/")
async def root() -> str:
    """Root endpoint returning a welcome message."""
    return function()


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    deployments = flyte.deploy(app_env)
    d = deployments[0]
    print(f"Deployed FastAPI app: {d.table_repr()}")
