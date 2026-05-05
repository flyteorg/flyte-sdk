import asyncio
import os
from pathlib import Path
from typing import List

import flyte
import flyte.storage
from flyte.storage import S3

env = flyte.TaskEnvironment(name="hello_world", cache="disable")


@env.task
async def say_hello_hybrid(data: str, lt: List[int]) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data} {lt}"


@env.task
async def squared(i: int = 3) -> int:
    print(flyte.ctx().action)
    return i * i


@env.task
async def squared_2(i: int = 3) -> int:
    print(flyte.ctx().action)
    return i * i


@env.task
async def say_hello_hybrid_nested(data: str = "default string") -> str:
    print(f"Hello, nested! - {flyte.ctx().action}")
    coros = []
    for i in range(3):
        coros.append(squared(i=i))

    vals = await asyncio.gather(*coros)
    return await say_hello_hybrid(data=data, lt=vals)


@env.task
async def hybrid_parent_placeholder():
    import sys
    import time

    print(f"Hello, hybrid parent placeholder - Task Context: {flyte.ctx()}")
    print(f"Run command: {sys.argv}")
    print("Environment Variables:")
    for k, value in sorted(os.environ.items()):
        if k.startswith("FLYTE_") or k.startswith("_U"):  # noqa: PIE810
            print(f"{k}: {value}")

    print("Sleeping for 24 hours to simulate a long-running task...", flush=True)
    time.sleep(86400)  # noqa: ASYNC251


if __name__ == "__main__":
    # Configurable via env vars so the example doesn't bake in machine-local paths.
    #   FLYTE_CONFIG       -> path to flyte config yaml (default: ~/.flyte/config.yaml)
    #   FLYTE_HYBRID_RUN_NAME -> the long-running parent run id to attach hybrid sub-actions to
    #   FLYTE_HYBRID_RUN_BASE -> s3 base dir for the hybrid run
    current_directory = Path(os.getcwd())
    repo_root = current_directory.parent.parent
    s3_sandbox = S3.for_sandbox()

    config_path = os.getenv("FLYTE_CONFIG") or str(Path.home() / ".flyte" / "config.yaml")
    flyte.init_from_config(config_path, root_dir=repo_root, storage=s3_sandbox)

    run_name = os.getenv("FLYTE_HYBRID_RUN_NAME")
    run_base_dir = os.getenv("FLYTE_HYBRID_RUN_BASE")
    if not run_name or not run_base_dir:
        raise SystemExit(
            "Set FLYTE_HYBRID_RUN_NAME and FLYTE_HYBRID_RUN_BASE to the parent run id "
            "and s3 base dir of an already-launched hybrid run before invoking this example."
        )

    outputs = flyte.with_runcontext(
        mode="hybrid",
        name=run_name,
        run_base_dir=run_base_dir,
    ).run(say_hello_hybrid_nested, data="hello world")
    print("Output:", outputs)
