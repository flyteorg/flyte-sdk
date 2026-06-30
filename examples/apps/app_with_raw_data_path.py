"""FastAPI app that returns the raw data path from flyte.app.ctx."""

from fastapi import FastAPI

import flyte
from flyte.app import ctx
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="Raw Data Path Demo",
    description="Returns the raw data path from flyte.app.ctx",
    version="1.0.0",
)

app_env = FastAPIAppEnvironment(
    name="raw-data-path-demo",
    app=app,
    description="App that returns the raw data path from ctx",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    env_vars={"LOG_LEVEL": "10"},
)


@app.get("/")
async def root() -> str:
    """Return the raw data path from flyte.app.ctx."""
    return ctx().raw_data_path or ""


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_handle = flyte.serve(app_env)
    print(f"Deployed app: {app_handle.url}")
