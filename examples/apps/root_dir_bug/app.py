# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "flyte==2.0.0b43"
# ]
# ///

"""
Demonstration of the `flyte serve` root dir autodetection bug.

DIRECTORY LAYOUT
----------------
examples/apps/
├── shared_utils.py          <-- shared utility, lives in apps/
└── root_dir_bug/
    └── app.py               <-- this script

HOW TO REPRODUCE
----------------
From the project root, run:

    flyte serve examples/apps/root_dir_bug/app.py env

Deployment will fail because `examples/apps/root_dir_bug/` is used as the
bundle root, making `../shared_utils.py` an invalid tar entry.
"""

import logging
import pathlib

from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(
    title="Root-dir Bug Demo",
    description="Demonstrates the flyte serve root dir autodetection bug",
    version="1.0.0",
)

# `../shared_utils.py` escapes this script's directory (root_dir_bug/) and
# references a file in the parent directory (apps/). This is the pattern that
# triggers the bug.
env = FastAPIAppEnvironment(
    name="root-dir-bug-demo",
    app=app,
    description="App that includes a file outside its own directory.",
    image=flyte.Image.from_uv_script(__file__, name="root-dir-bug-script"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    include=["../shared_utils.py"],  # <-- escapes root_dir_bug/, triggers the bug
)


@env.app.get("/")
async def root() -> dict[str, str]:
    from examples.apps.shared_utils import greet

    return {"message": greet("World")}


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent.parent,  # examples/apps/
        log_level=logging.DEBUG,
    )
    deployments = flyte.deploy(env)
    d = deployments[0]
    print(f"Deployed app: {d.table_repr()}")
