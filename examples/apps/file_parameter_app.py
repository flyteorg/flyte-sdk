"""
File Parameter App
==================

Single-file example showing how to:
1. Run a Flyte task that produces a remote file (flyte.io.File).
2. Feed that file into a FastAPI app using flyte.app.Parameter with
   flyte.io.File.from_existing_remote.

Usage:
    python file_parameter_app.py
"""

import logging
import os
import pathlib

from fastapi import FastAPI

import flyte
import flyte.io
from flyte.app import Parameter, get_parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared image
# ---------------------------------------------------------------------------

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn")

# ---------------------------------------------------------------------------
# Task: produce a remote file
# ---------------------------------------------------------------------------

task_env = flyte.TaskEnvironment(
    name="file-producer",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

FILE_NAME = "produced_file.txt"


@task_env.task
async def produce_file() -> flyte.io.File:
    """Write a greeting to a new remote file and return it."""
    f = flyte.io.File.new_remote(file_name=FILE_NAME)
    async with f.open("wb") as fh:
        await fh.write(b"Hello from the produce_file task!")
    logger.info("Wrote file to %s", f.path)
    return f


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

DATA_MOUNT_PATH = "/tmp/produced_file_new.txt"
DATA_ENV_VAR = "DATA_FILE_PATH"

DATA_DIR_MOUNT_PATH = "/tmp/"
DATA_DIR_ENV_VAR = "DATA_DIR_PATH"

app = FastAPI(
    title="File Parameter Demo",
    description="Serves content from a file produced by a Flyte task",
    version="1.0.0",
)

app_env = FastAPIAppEnvironment(
    name="file-parameter-demo",
    app=app,
    description="Serves content from a task-produced file",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    parameters=[
        Parameter(name="data", type="file", mount=DATA_MOUNT_PATH, env_var=DATA_ENV_VAR),
        Parameter(name="data_with_dir", type="file", mount=DATA_DIR_MOUNT_PATH, env_var=DATA_DIR_ENV_VAR),
        Parameter(name="data_raw", type="file"),
    ],
    env_vars={"LOG_LEVEL": "10"},
)


@app.get("/")
def root(from_dir: bool = False) -> dict:
    """Return the contents of the parameter file."""
    data_path = DATA_DIR_MOUNT_PATH + FILE_NAME if from_dir else DATA_MOUNT_PATH
    if not os.path.exists(data_path):
        return {"error": "data file not available", "path": data_path}
    with open(data_path, "rb") as fh:
        contents = fh.read().decode("utf-8")
    return {"contents": contents, "path": data_path, "mount_path": data_path, "from_dir": from_dir}


@app.get("/from-env-var")
def from_env_var(from_dir: bool = False) -> dict:
    """Return the contents of the parameter file."""
    data_path = os.environ.get(DATA_DIR_ENV_VAR) if from_dir else os.environ.get(DATA_ENV_VAR)
    if not os.path.exists(data_path):
        return {"error": "data file not available", "path": data_path}
    with open(data_path, "rb") as fh:
        contents = fh.read().decode("utf-8")
    return {"contents": contents, "path": data_path, "env_var": data_path, "from_dir": from_dir}


@app.get("/from-helper-function")
def from_helper_function(from_dir: bool = False) -> dict:
    """Return the contents of the parameter file."""
    parameter_name = "data_with_dir" if from_dir else "data"
    data_path = get_parameter(parameter_name)
    if not os.path.exists(data_path):
        return {"error": "data file not available", "path": data_path}
    with open(data_path, "rb") as fh:
        contents = fh.read().decode("utf-8")
    return {"contents": contents, "path": data_path, "parameter": parameter_name, "from_dir": from_dir}


@app.get("/raw")
def raw() -> dict:
    """Return the contents of the parameter file."""
    data_path = get_parameter("data_raw")
    if not os.path.exists(data_path):
        return {"error": "data file not available", "path": data_path}
    with open(data_path, "rb") as fh:
        contents = fh.read().decode("utf-8")
    return {"contents": contents, "path": data_path, "parameter": "data_raw"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main: run task -> create app env with File.from_existing_remote -> deploy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(
        # root_dir=pathlib.Path(__file__).parent,
        root_dir=pathlib.Path.cwd(),
        log_level=logging.DEBUG,
    )

    # Step 1: Run the task that produces a remote file.
    run = flyte.run(produce_file)
    print(f"Task run: {run.url}")
    run.wait()

    # Step 2: Retrieve the output file's remote path.
    output_file: flyte.io.File = run.outputs()[0]
    remote_path = output_file.path
    print(f"Produced file at: {remote_path}")

    # Step 4: Deploy the app.
    app_handle = flyte.with_servecontext(
        parameter_values={
            app_env.name: {
                "data": flyte.io.File.from_existing_remote(remote_path),
                "data_with_dir": flyte.io.File.from_existing_remote(remote_path),
                "data_raw": flyte.io.File.from_existing_remote(remote_path),
            }
        }
    ).serve(app_env)
    print(f"Deployed app: {app_handle.url}")
