"""
Code Sandbox — Running Arbitrary Code in Isolated Containers
============================================================

``flyte.sandbox.create()`` runs arbitrary Python code or shell commands inside
an ephemeral, **stateless** Docker container. The container is built on demand
from the declared ``packages`` / ``system_packages``, executed once, and
discarded. Each invocation starts from a clean slate — no filesystem state,
environment variables, or side effects carry over between runs.

Three modes:

- **Auto-IO mode** (``code=``, ``auto_io=True``, default): write just the business logic.
  Flyte auto-generates an argparse preamble so declared inputs are available as
  local variables, and writes scalar outputs automatically. No boilerplate.
- **Verbatim mode** (``code=``, ``auto_io=False``): run a complete Python script as-is.
  CLI args are forwarded but the script handles all I/O itself.
- **Command mode** (``command=``): run any shell command (pytest, binary, non-Python code, etc.).
"""

import asyncio
import datetime
from pathlib import Path

import nest_asyncio

import flyte
import flyte.sandbox
from flyte._image import PythonWheels
from flyte.io import Dir, File
from flyte.sandbox import sandbox_environment

nest_asyncio.apply()


env = flyte.TaskEnvironment(
    name="sandbox-demo",
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
            ),
            name="sandbox-demo-image",
        )
        .with_pip_packages("nest-asyncio")
    ),
    depends_on=[sandbox_environment],
)


# Example 1 — code mode (auto-inject): pure Python, no boilerplate
#
# Declare inputs/outputs and write just the business logic.
# ``n`` is available as a local variable; ``total`` is collected automatically.

sum_sandbox = flyte.sandbox.create(
    name="sum-to-n",
    code="total = sum(range(n + 1)) if conditional else 0",
    inputs={"n": int, "conditional": bool},
    outputs={"total": int},
    cache="disable",
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
    cache="disable",
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

# Example 5 — Dir input/output support
#
# ``Dir`` inputs arrive as local directory paths in auto-IO mode. To return a
# ``Dir``, create the directory at ``/var/outputs/<name>`` and write files
# inside it.

_dir_transform_code = """\
import pathlib

src = pathlib.Path(source_dir)
dst = pathlib.Path("/var/outputs/processed_dir")
dst.mkdir()

copied_files = 0
for path in sorted(src.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(src)
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(path.read_text().upper())
    copied_files += 1

(dst / "summary.txt").write_text(f"copied_files={copied_files}\\n")
"""

dir_sandbox = flyte.sandbox.create(
    name="dir-transform",
    code=_dir_transform_code,
    inputs={"source_dir": Dir},
    outputs={"processed_dir": Dir, "copied_files": int},
)

# Example 6 — block_network=True: no outbound network access
#
# Network access is blocked by default. The sandbox can only read its inputs
# and write its outputs — any attempt to reach the network will fail.

isolated_sandbox = flyte.sandbox.create(
    name="isolated-compute",
    code="""\
import urllib.request
with urllib.request.urlopen("https://example.com") as r:
    status = str(r.status)
""",
    outputs={"status": str},
    block_network=True,
)

# Example 7 — block_network=False (default): outbound network access allowed

networked_sandbox = flyte.sandbox.create(
    name="networked-fetch",
    code="""\
import urllib.request
with urllib.request.urlopen("https://example.com") as r:
    status = str(r.status)
""",
    outputs={"status": str},
)


# Example 8 — as_task() + deploy: build a deployable sandbox task
#
# Use ``as_task()`` to get a ContainerTask with ``_script`` pre-filled as a
# default.


@env.task
async def deploy_sandbox_task() -> str:
    """Deploy a sandbox task using as_task().

    Builds the image, creates a ContainerTask with ``_script`` pre-filled as a
    default value, and deploys it. Retriggers from the UI only need user inputs.
    """
    flyte.init_in_cluster()

    sandbox = flyte.sandbox.create(
        name="deployable-sandbox",
        code="""\
import json, pathlib
data = json.loads(pathlib.Path(payload).read_text())
total = sum(data["values"])
""",
        inputs={"payload": File},
        outputs={"total": int},
        resources=flyte.Resources(cpu=1, memory="512Mi"),
    )
    task = await sandbox.as_task.aio()
    deploy_env = flyte.TaskEnvironment.from_task("deployable-sandbox", task)
    v = flyte.deploy(deploy_env)

    print("Deployed environment:", v[0].summary_repr())
    return v[0].summary_repr()


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
async def create_input_dir() -> Dir:
    """Create a small directory tree for the Dir sandbox example."""
    base = Path("/tmp/sandbox-dir-input")
    base.mkdir(parents=True, exist_ok=True)
    (base / "a.txt").write_text("alpha\n")
    nested = base / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "b.txt").write_text("beta\n")
    return await Dir.from_local(str(base))


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

    # Dir input/output mode: transform an input directory into an output directory
    input_dir = await create_input_dir()
    processed_dir, copied_files = await dir_sandbox.run.aio(source_dir=input_dir)
    processed_dir_path = Path(await processed_dir.download())
    dir_output_files = sorted(
        str(path.relative_to(processed_dir_path)) for path in processed_dir_path.rglob("*") if path.is_file()
    )
    summary_text = (processed_dir_path / "summary.txt").read_text().strip()

    # Network-blocked sandbox: pure computation, no network
    try:
        isolated_result = await isolated_sandbox.run.aio()
    except Exception as e:
        isolated_result = f"Failed to access network: {e}"

    # Network-allowed sandbox: fetch external URL
    status = await networked_sandbox.run.aio()

    # Deploy a sandbox task
    deployment_summary = await deploy_sandbox_task()

    return {
        "sum_1_to_10": total,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "window_end": window_end.isoformat(),
        "etl_sum_1_to_10": etl_total,
        "line_count": line_count,
        "dir_copied_files": copied_files,
        "dir_output_files": ",".join(dir_output_files),
        "dir_summary": summary_text,
        "isolated_result": isolated_result,
        "networked_status": status,
        "deployment_summary": deployment_summary,
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
    input_dir = asyncio.run(create_input_dir())
    processed_dir, copied_files = dir_sandbox.run(source_dir=input_dir)
    processed_dir_path = Path(processed_dir.download_sync())
    dir_output_files = sorted(
        str(path.relative_to(processed_dir_path)) for path in processed_dir_path.rglob("*") if path.is_file()
    )
    summary_text = (processed_dir_path / "summary.txt").read_text().strip()

    try:
        isolated_result = isolated_sandbox.run()
    except Exception as e:
        isolated_result = f"Failed to access network: {e}"

    status = networked_sandbox.run()
    deployment_summary = asyncio.run(deploy_sandbox_task())

    return {
        "sum_1_to_10": total,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "window_end": window_end.isoformat(),
        "etl_sum_1_to_10": etl_total,
        "line_count": line_count,
        "dir_copied_files": copied_files,
        "dir_output_files": ",".join(dir_output_files),
        "dir_summary": summary_text,
        "isolated_result": isolated_result,
        "networked_status": status,
        "deployment_summary": deployment_summary,
    }


if __name__ == "__main__":
    flyte.init_from_config()

    run_async = flyte.with_runcontext(mode="remote").run(run_pipeline)
    run_sync = flyte.with_runcontext(mode="remote").run(run_pipeline_sync)

    print("Async run URL:", run_async.url)
    print("Sync run URL:", run_sync.url)
