# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte",
#    "fastapi",
#    "flyteplugins-union>=0.19.0",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Serve a Volume from an app.

A task builds a Volume and seals it into an immutable ``ROVolume``; an app
mounts that version read-only and serves its files. Apps take the same
``pod_template=allow_volumes()`` as tasks — the mount goes through the node
mount broker, so the app pod needs no CAP_SYS_ADMIN, no /dev/fuse, and no
hostPath.

The version is handed to the app as its ``locator`` (a plain string) and
reloaded with ``Volume.from_locator``.

Prereqs: the Union mount broker on the cluster (``uvolMountBroker``), and
Knative Serving with ``kubernetes.podspec-volumes-csi`` and
``kubernetes.podspec-volumes-mount-propagation`` enabled (the dataplane
chart's vendored gateway enables both).

Usage:
    python volume_app.py
"""

import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from flyteplugins.union.io import Volume, allow_volumes

import flyte
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCATOR_ENV_VAR = "VOLUME_LOCATOR"

image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-app")
    .with_pip_packages("fastapi", "uvicorn", "flyteplugins-union>=0.19.0")
    .with_local_v2()  # last, so the local dev SDK wins over the pip layer's flyte dep
)

task_env = flyte.TaskEnvironment(
    name="volume-app-producer",
    image=image,
    pod_template=allow_volumes(),
    resources=flyte.Resources(cpu=1, memory="4Gi"),
)


@task_env.task
async def build_volume() -> str:
    """Write a few files into a new Volume, seal it, and return its locator."""
    vol = Volume.new(name="volume-app-demo")
    root = pathlib.Path(await vol.mount())
    (root / "hello.txt").write_text("Hello from a Union Volume!\n")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# Served read-only by an app\n")
    sealed = await vol.finalize()
    logger.info("Sealed volume at %s", sealed.locator)
    return sealed.locator


@asynccontextmanager
async def lifespan(app: FastAPI):
    vol = await Volume.from_locator(os.environ[LOCATOR_ENV_VAR])
    app.state.mount_root = pathlib.Path(await vol.mount())
    logger.info("Mounted %s at %s", vol.name, app.state.mount_root)
    yield


app = FastAPI(title="Volume App", lifespan=lifespan)

app_env = FastAPIAppEnvironment(
    name="volume-app",
    app=app,
    image=image,
    # Same helper as tasks; the app's primary container is named for you.
    pod_template=allow_volumes(),
    resources=flyte.Resources(cpu=1, memory="4Gi"),
    parameters=[Parameter(name="locator", type="string", env_var=LOCATOR_ENV_VAR)],
    requires_auth=False,
)


@app.get("/files/{path:path}")
async def files(path: str = ""):
    mount_root = app.state.mount_root
    target = (mount_root / path).resolve()
    if not target.is_relative_to(mount_root) or not target.exists():
        raise HTTPException(status_code=404)
    if target.is_dir():
        # Skip JuiceFS's virtual control files (.accesslog, .stats, ...) at the root.
        return sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir() if not p.name.startswith("."))
    return {"path": path, "content": target.read_text()}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mount_root": str(app.state.mount_root)}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

    run = flyte.run(build_volume)
    print(f"Task run: {run.url}")
    run.wait()
    locator = run.outputs()[0]

    app_handle = flyte.with_servecontext(parameter_values={app_env.name: {"locator": locator}}).serve(app_env)
    print(f"Deployed app: {app_handle.url}")
