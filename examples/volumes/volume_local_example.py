# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Local-backend Volume example.

Mirrors ``volume_example.py`` but uses ``volume_backend="local"`` — a
tar.gz archive in object storage, no FUSE, no privileged container.
Runs the same ``init → append → fork → append (fork only) → read``
lineage and verifies that the fork sees the parent's bytes.

The same workflow works under ``flyte run --local``: the underlying
``flyte.storage`` layer maps remote URIs to local paths transparently,
so the tar archive lives next to the workflow's other intermediates.
"""

import logging
import os
import uuid
from pathlib import Path

import flyte
from flyte.extras import Volume, volume_image_local, volume_pod_template_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-local-demo")

VOL_NAME = os.environ.get("VOL_NAME", "local-demo-vol")

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
base = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-local-demo")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = volume_image_local(base).with_local_v2()

env = flyte.TaskEnvironment(
    name="volume-local-demo",
    pod_template=volume_pod_template_local(),
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)


@env.task(cache="auto")
async def init_volume(volume_name: str) -> Volume:
    logger.info("init_volume: declaring fresh local volume name=%s", volume_name)
    vol = Volume.empty(name=volume_name, volume_backend="local")
    await vol.mount()
    Path("/workspace/hello.txt").write_text("hello from local volume\n")
    committed = await vol.commit()
    logger.info("init_volume: committed index=%s", committed.index.path if committed.index else None)
    return committed


@env.task
async def append(vol: Volume, line: str) -> Volume:
    logger.info("append: mounting volume name=%s", vol.name)
    await vol.mount()
    with open("/workspace/hello.txt", "a") as f:  # noqa: ASYNC230
        f.write(line + "\n")
    return await vol.commit()


@env.task
async def branch(vol: Volume, name: str) -> Volume:
    logger.info("branch: cold-forking name=%s -> %s (no parent mount)", vol.name, name)
    return await vol.fork(name=name)


@env.task
async def read_all(vol: Volume) -> str:
    logger.info("read_all: mounting volume name=%s", vol.name)
    await vol.mount()
    return Path("/workspace/hello.txt").read_text()


@env.task
async def main() -> str:
    run_name = f"{VOL_NAME}-{uuid.uuid4().hex[:8]}"
    v0 = await init_volume(volume_name=run_name)
    v1 = await append(v0, line="appended on main branch")
    fork = await branch(v1, name=f"{run_name}-fork")
    fork_after = await append(fork, line="only on the fork")
    return await read_all(fork_after)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
