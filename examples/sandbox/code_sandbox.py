"""
Code Sandbox — Running Arbitrary Code in Isolated Containers
============================================================

``flyte.sandbox.create()`` runs arbitrary Python code or shell commands inside
an ephemeral Docker container. The container is built on demand from the
declared ``packages`` / ``system_packages``, executed once, and discarded.

Three modes:

- **Auto-IO mode** (``code=``, ``auto_io=True``, default): write just the business logic.
  Flyte auto-generates an argparse preamble so declared inputs are available as
  local variables, and writes scalar outputs automatically. No boilerplate.
- **Verbatim mode** (``code=``, ``auto_io=False``): run a complete Python script as-is.
  No CLI args injected — the script handles all I/O itself.
- **Command mode** (``command=``): run any shell command (pytest, binary, non-Python code, etc.).

Security defaults:
- ``block_network=True`` (default) — the container has *no network interface*
  (``--network none``), preventing any outbound calls from untrusted code.
"""

import asyncio
import datetime
from pathlib import Path

import nest_asyncio

import flyte
import flyte.sandbox
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

nest_asyncio.apply()


# Example 1 — code mode (auto-inject): pure Python, no boilerplate
#
# Declare inputs/outputs and write just the business logic.
# ``n`` is available as a local variable; ``total`` is collected automatically.

sum_sandbox = flyte.sandbox.create(
    name="sum-to-n",
    code="total = sum(range(n + 1)) if conditional else 0",
    inputs={"n": int, "conditional": bool},
    outputs={"total": int},
)

# Example 2 — code mode (auto-inject): third-party packages (numpy)
#
# Just the computation — no argparse, no output-file writing.

_stats_code = """\
import numpy as np
nums = np.array([float(v) for v in values.split(",")])
mean = float(np.mean(nums))
std  = float(np.std(nums))

window_end = dt + delta
"""

stats_sandbox = flyte.sandbox.create(
    name="numpy-stats",
    code=_stats_code,
    inputs={
        "values": str,
        "dt": datetime.datetime,
        "delta": datetime.timedelta,
    },
    outputs={"mean": float, "std": float, "window_end": datetime.datetime},
    packages=["numpy"],
)

# Example 3 — verbatim mode: complete Python script, full control
#
# The user handles all I/O themselves — Flyte just runs ``python script.py``
# with no injected CLI args. File inputs are bind-mounted at /var/inputs/<name>.

_etl_script = """\
import json, pathlib

payload = json.loads(pathlib.Path("/var/inputs/payload").read_text())
total = sum(payload["values"])

pathlib.Path("/var/outputs/total").write_text(str(total))
"""

etl_sandbox = flyte.sandbox.create(
    name="etl-script",
    code=_etl_script,
    inputs={"payload": File},
    outputs={"total": int},
    auto_io=False,
)

# Example 4 — command mode: shell pipeline

linecount_sandbox = flyte.sandbox.create(
    name="line-counter",
    command=[
        "/bin/bash",
        "-c",
        "grep -c . /var/inputs/data_file > /var/outputs/line_count || echo 0 > /var/outputs/line_count",
    ],
    inputs={"data_file": File},
    outputs={"line_count": str},
)

env = flyte.TaskEnvironment(
    name="sandbox-demo",
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"),
            name="sandbox-demo-image",
        )
        .with_pip_packages("nest-asyncio")
    ),
    depends_on=[sandbox_environment],
)


@env.task
async def create_text_file() -> File:
    """Create a small text file and return it as a File handle."""
    path = Path("/tmp/data.txt")
    path.write_text("line 1\n\nline 2\n")
    return await File.from_local(str(path))


@env.task
async def payload_generator() -> File:
    """Generate a JSON payload file for the ETL example."""
    path = Path("/tmp/payload.json")
    path.write_text('{"values": [1, 2, 3, 4, 5]}')
    return await File.from_local(str(path))


@env.task
async def run_pipeline() -> dict:
    """Run all sandbox examples and return their outputs."""
    # Auto-inject: sum 1..10 = 55
    total = await sum_sandbox.run.aio(n=10, conditional=True)

    # Auto-inject with packages: numpy stats
    mean, std, window_end = await stats_sandbox.run.aio(
        values="1,2,3,4,5",
        dt=datetime.datetime(2024, 1, 1),
        delta=datetime.timedelta(days=1),
    )

    # ETL script: sum 1..10 = 55
    payload = await payload_generator()
    etl_total = await etl_sandbox.run.aio(payload=payload)

    # Command mode: line count
    data_file = await create_text_file()
    line_count = await linecount_sandbox.run.aio(data_file=data_file)

    return {
        "sum_1_to_10": total,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "window_end": window_end.isoformat(),
        "etl_sum_1_to_10": etl_total,
        "line_count": line_count,
    }


@env.task
def run_pipeline_sync() -> dict:
    """Sync version of the above."""
    total = sum_sandbox.run(n=10, conditional=True)
    mean, std, window_end = stats_sandbox.run(
        values="1,2,3,4,5",
        dt=datetime.datetime(2024, 1, 1),
        delta=datetime.timedelta(days=1),
    )
    payload = asyncio.run(payload_generator())
    etl_total = etl_sandbox.run(payload=payload)
    data_file = asyncio.run(create_text_file())
    line_count = linecount_sandbox.run(data_file=data_file)
    return {
        "sum_1_to_10": total,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "window_end": window_end.isoformat(),
        "etl_sum_1_to_10": etl_total,
        "line_count": line_count,
    }


if __name__ == "__main__":
    flyte.init_from_config()

    run_async = flyte.with_runcontext(mode="remote").run(run_pipeline)
    run_sync = flyte.with_runcontext(mode="remote").run(run_pipeline_sync)

    print("Async run URL:", run_async.url)
    print("Sync run URL:", run_sync.url)
