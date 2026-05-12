"""Basic shell task — File in, scalar in, File out.

The minimal shape: one input file, one scalar parameter, one output file.
Run locally with::

    uv run python 01_basic.py
"""

import sys
import tempfile

import flyte
from flyte.extras import shell
from flyte.io import File
from flyte._image import PythonWheels
from pathlib import Path

# Wrap `head` — emits the first N lines of an input file.
head_task = shell.create(
    name="head",
    image="debian:12-slim",
    inputs={
        "src": File,  # input file (mounted at /var/inputs/src)
        "n": int,  # number of lines to emit
    },
    outputs={
        "out": File,
    },
    script=r"""
        head -n {inputs.n} {inputs.src} > {outputs.out}
    """,
)


env = flyte.TaskEnvironment(
    name="shell_basic",
    depends_on=[head_task.env],
    image=(
        flyte.Image.from_debian_base().clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
            ),
            name="shell-basic",
        )
    ),
)


@env.task
async def take_first_lines(src: File, n: int) -> File:
    return await head_task(src=src, n=n)


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(f"line {i}" for i in range(1, 21)))
        path = f.name

    run = flyte.with_runcontext(mode=mode).run(
        take_first_lines, File.from_local_sync(path), 5
    )
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
