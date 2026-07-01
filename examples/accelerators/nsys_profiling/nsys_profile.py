"""Profile a GPU training script with Nsight Systems (`nsys`) in a Flyte task.

Runs::

    nsys profile --trace=cuda,nvtx,osrt \
        --output=lightning_nsys_trace --force-overwrite=true \
        python train.py

then:

- uploads the resulting ``lightning_nsys_trace.nsys-rep`` as a task **output**
  (a ``flyte.io.File`` you can download from the run and open in the Nsight
  Systems GUI), and
- runs ``nsys stats`` on it and renders the summary tables into the task's
  **HTML report deck** (``report=True`` + ``flyte.report``).

``train.py`` ships as a real file via ``include=`` (not an inline string), so you
can edit/run it on its own and drop in your own training script.

Why the pod template matters
----------------------------
``--trace=osrt`` (OS-runtime tracing) and GPU-counter sampling need
``CAP_SYS_ADMIN`` + an unconfined AppArmor profile, or profiling silently yields
an empty trace / ``ERR_NVGPUCTRPERM``. ``PodTemplate.allow_nested_sandboxing()``
grants exactly that bundle. The cluster-clean alternative is to set
``NVreg_RestrictProfilingToAdminUsers=0`` on the GPU nodes' driver and drop the
cap.

Run::

    flyte run examples/accelerators/nsys_profiling/nsys_profile.py profile_training
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import flyte
import flyte.report
from flyte.io import File

_HERE = Path(__file__).parent

# NGC PyTorch container ships torch **and** the Nsight Systems CLI (`nsys`)
# preinstalled, so there's no apt/CUDA-repo dance. It pulls anonymously.
#
# Making an arbitrary base image flyte-runnable takes a specific bundle:
#   - `.clone(extendable=True, name=...)` — `from_base` images are non-extendable
#     and unnamed by default; adding any layer otherwise raises "Cannot add
#     additional layers to a non-extendable image".
#   - `python_version=(3, 10)` — match NGC 24.08's interpreter, else the builder
#     runs `uv venv --python <your-local-version>` and fails ("No interpreter
#     found").
#   - `flyte`      — the task bootstrap imports the SDK; the raw NGC image has none.
#   - `uv`         — flyte's entrypoint runs `uv run ...` and NGC has no `uv`.
#   - `kubernetes` — this module builds `allow_nested_sandboxing()` at import
#     time, which imports `kubernetes.client`.
# Adding these layers also forces flyte to build a proper image with its own
# entrypoint; the bare base's `nvidia_entrypoint.sh` otherwise hijacks startup.
#
# flyte installs the above into an isolated venv at `/opt/venv`, which does NOT
# see NGC's system-installed torch. We deliberately DON'T reinstall torch there
# (heavy + risks clashing with NGC's tuned CUDA libs); the task instead profiles
# with NGC's own torch-capable interpreter (see `_torch_python`).
image = (
    flyte.Image.from_base("nvcr.io/nvidia/pytorch:24.08-py3")
    .clone(extendable=True, name="nsys-profile", python_version=(3, 10))
    .with_pip_packages("flyte", "uv", "kubernetes")
)

env = flyte.TaskEnvironment(
    name="nsys-profiling",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1"),
    # CAP_SYS_ADMIN + AppArmor-unconfined — required for osrt tracing / GPU counters.
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
    # Bundle train.py alongside the code so the task can profile it.
    include=[str(_HERE / "train.py")],
)

_REPORT_NAME = "lightning_nsys_trace"  # -> lightning_nsys_trace.nsys-rep


async def _run(*cmd: str, cwd: str) -> tuple[int, str, str]:
    """Run a subprocess, capturing stdout/stderr as text."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _torch_python() -> str:
    """Return an interpreter that can `import torch`.

    The flyte task runs inside `/opt/venv` (flyte only). NGC's torch lives in the
    system interpreter, so we pick the first candidate that actually imports
    torch instead of the venv `python`.
    """
    import shutil
    import subprocess

    candidates = [
        "/usr/bin/python3.10", "/usr/bin/python3", "/usr/local/bin/python",
        shutil.which("python3") or "", shutil.which("python") or "",
    ]
    tried = []
    for exe in candidates:
        if not exe or not os.path.exists(exe):
            continue
        tried.append(exe)
        if subprocess.run([exe, "-c", "import torch"], capture_output=True).returncode == 0:
            return exe
    raise RuntimeError(f"no interpreter with torch found (tried: {tried})")


@env.task(report=True)
async def profile_training() -> File:
    """Profile `train.py` under nsys and return the .nsys-rep file."""
    workdir = "/tmp/nsys"
    os.makedirs(workdir, exist_ok=True)
    train_py = _HERE / "train.py"

    py = _torch_python()
    print("profiling with interpreter:", py)

    # nsys profile --trace=cuda,nvtx,osrt --output=... --force-overwrite=true <py> train.py
    rc, out, err = await _run(
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        f"--output={_REPORT_NAME}",
        "--force-overwrite=true",
        py, str(train_py),
        cwd=workdir,
    )
    print(out)
    if err:
        print("nsys stderr:\n", err)
    if rc != 0:
        raise RuntimeError(f"nsys profile failed (rc={rc}):\n{err[-2000:]}")

    report_path = os.path.join(workdir, f"{_REPORT_NAME}.nsys-rep")
    if not os.path.exists(report_path):
        raise RuntimeError(f"expected report not produced at {report_path}")

    # Summarize into the HTML report deck (viewable in the console).
    _, stats_out, stats_err = await _run(
        "nsys", "stats",
        "--report", "cuda_gpu_kern_sum",
        "--report", "nvtx_sum",
        "--report", "osrt_sum",
        f"{_REPORT_NAME}.nsys-rep",
        cwd=workdir,
    )
    await flyte.report.log.aio(
        "<h2>nsys stats — kernel / NVTX / OS-runtime summary</h2>"
        f"<pre>{(stats_out or stats_err) or '(no summary)'}</pre>"
        "<p>Full timeline: download the <code>.nsys-rep</code> output and open "
        "it in the Nsight Systems GUI.</p>",
        do_flush=True,
    )

    # Return the report as a first-class File output (uploaded to blob storage).
    return await File.from_local(report_path)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(profile_training)
    print(run.url)
