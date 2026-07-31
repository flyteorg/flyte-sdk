"""Low-level control of the NVIDIA Nsight Systems CLI (`nsys`).

The task process is launched under `nsys launch --session-new=<session> --trace=<trace>` by the
runtime re-exec hook (see the _F_EXEC_WRAPPER handling in flyte._bin.runtime). Launch is where
nsys takes the trace/sample configuration; it establishes an interactive session but does not
collect anything yet. The helpers here drive that session from inside the still-running task:

    nsys start --session=<session> -o <output>   # begin collecting (trace was set at launch)
    ...the code we want to profile runs...
    nsys stop  --session=<session>                # finalize <output>.nsys-rep

`nsys stop` flushes the report to disk while the task keeps running, so the task can then
summarize it and hand it back as a trace output. None of this is visible to the user, who
only adds a decorator.

Everything here is best-effort: a profiler that cannot attach, or a cluster that withholds
the required capabilities, must never take down the real work. Failures are logged and the
task continues unprofiled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Set by the nsys_profile decorator onto the task container, read back here at runtime.
# _F_EXEC_WRAPPER / _F_EXEC_WRAPPED are the generic runtime re-exec contract; the rest are ours.
# Trace and sample are baked into the ENV_WRAPPER launch command (the only place nsys accepts
# them), so they need no env vars of their own.
ENV_WRAPPER = "_F_EXEC_WRAPPER"
ENV_WRAPPED = "_F_EXEC_WRAPPED"
# Opts the runtime re-exec hook into wrapping the primary worker of a clustered (torchrun) task; set
# only for clustered tasks. Part of the generic re-exec contract, read by flyte._bin.runtime.
ENV_CLUSTERED = "_F_EXEC_WRAPPER_CLUSTERED"
ENV_SESSION = "_F_NSYS_SESSION"
ENV_OUTPUT = "_F_NSYS_OUTPUT"

# `nsys start` returns before its collection is actually capturing. Settle this long after a
# successful start so a short region's kernels aren't missed while the session arms.
_ARM_SETTLE_SEC = 0.5

# Hints in nsys stderr that the pod lacks the capabilities profiling needs. Mapped to an
# actionable message instead of a raw driver error.
_PERM_HINTS = ("ERR_NVGPUCTRPERM", "insufficient privileges", "CAP_SYS_ADMIN", "capability")

_PERM_REMEDY = (
    "nsys could not access GPU performance counters / OS-runtime tracing. This needs elevated "
    "pod capabilities. Either give the task `pod_template=flyte.PodTemplate().allow_nested_sandboxing()` "
    "(adds CAP_SYS_ADMIN + unconfined AppArmor), or set NVreg_RestrictProfilingToAdminUsers=0 on the "
    "GPU nodes' driver. You can also restrict `trace` to cuda,nvtx which needs no extra capabilities."
)


class NsysError(RuntimeError):
    """A nsys CLI invocation failed. Callers treat this as non-fatal to the task."""


@dataclass(frozen=True)
class Collection:
    """A finished collection: where the report landed and how to label it."""

    name: str
    report_path: str


def nsys_available() -> bool:
    """True if the `nsys` binary is on PATH in this container."""
    return shutil.which("nsys") is not None


def under_nsys() -> bool:
    """True only when the runtime actually re-exec'd this process under `nsys launch`.

    Requires both the generic wrapped-flag and an nsys session name, so a plain local run
    (where the re-exec hook never fires) reports False and the task runs unprofiled.
    """
    return bool(os.environ.get(ENV_WRAPPED)) and bool(os.environ.get(ENV_SESSION))


def session_name() -> str:
    return os.path.expandvars(os.environ.get(ENV_SESSION, ""))


def _output_base(name: Optional[str]) -> str:
    base = os.path.expandvars(os.environ.get(ENV_OUTPUT, "/tmp/nsys/report"))
    if name:
        # A named region writes beside the base so several regions in one task don't collide.
        parent = os.path.dirname(base) or "."
        return os.path.join(parent, _safe(name))
    return base


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "region"


def build_container_env(
    *,
    session_template: str,
    output_template: str,
    trace: Sequence[str],
    sample: Optional[str],
    clustered: bool = False,
) -> dict[str, str]:
    """Build the env vars the decorator stamps onto the task container.

    ENV_WRAPPER is the generic instruction the runtime obeys to re-exec under `nsys launch`.
    Trace and sample are configured on that launch command, because nsys rejects --trace/--sample
    on `nsys start`. The templates keep $ACTION_NAME/$RUN_NAME unexpanded so they resolve per action
    in-container; trace/sample are comma-joined literals with no whitespace, so they survive the
    runtime's shlex.split of the wrapper as single tokens.

    clustered marks a torchrun task: it sets ENV_CLUSTERED so the runtime wraps only the primary
    worker (global RANK 0) under nsys and leaves the other ranks unprofiled. Only that one rank
    collects, so a single session/output is enough.
    """
    wrapper = f"nsys launch --session-new={session_template} --trace={','.join(trace)}"
    if sample:
        wrapper += f" --sample={sample}"
    env = {
        ENV_WRAPPER: wrapper,
        ENV_SESSION: session_template,
        ENV_OUTPUT: output_template,
    }
    if clustered:
        env[ENV_CLUSTERED] = "1"
    return env


async def _run(*cmd: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _run_sync(*cmd: str) -> tuple[int, str, str]:
    """Blocking twin of _run, for the sync collection path (a `with nsys.range(...)` block)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")


def _check_perms(stderr: str) -> None:
    if any(h in stderr for h in _PERM_HINTS):
        logger.warning(_PERM_REMEDY)


def _start_cmd(name: Optional[str]) -> tuple[list[str], str]:
    """Build the `nsys start` command and the report path it will write. Shared by the async and
    sync starters so both drive collection identically and differ only in how they run/sleep.

    Trace/sample were configured on `nsys launch`; start only controls where this collection writes.
    """
    session = session_name()
    if not session:
        raise NsysError("no nsys session in environment; task was not launched under nsys")

    output = _output_base(name)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    cmd = ["nsys", "start", f"--session={session}", "-o", output, "--force-overwrite=true"]
    return cmd, f"{output}.nsys-rep"


async def start_collection(name: Optional[str] = None) -> str:
    """Begin collecting on the current session. Returns the report path stop will write.

    Retries briefly: `nsys launch` may not have registered the session by the time the task reaches
    its first collection. And `nsys start` returns before the session is actually capturing, so a
    short region can run its kernels before collection arms and get a partial or empty trace (a long
    region hides this) — settle briefly after a successful start so collection is live first.
    """
    cmd, report_path = _start_cmd(name)
    last_err = ""
    for attempt in range(5):
        rc, out, err = await _run(*cmd)
        if rc == 0:
            await asyncio.sleep(_ARM_SETTLE_SEC)  # let collection actually begin before the body runs
            logger.info("nsys collection started -> %s", report_path)
            return report_path
        last_err = err or out
        _check_perms(last_err)
        await asyncio.sleep(0.5 * (attempt + 1))
    raise NsysError(f"`nsys start` failed after retries: {last_err[-800:]}")


def start_collection_sync(name: Optional[str] = None) -> str:
    """Blocking twin of start_collection, for a `with nsys.range(...)` block in a non-async task body."""
    cmd, report_path = _start_cmd(name)
    last_err = ""
    for attempt in range(5):
        rc, out, err = _run_sync(*cmd)
        if rc == 0:
            time.sleep(_ARM_SETTLE_SEC)  # let collection actually begin before the body runs
            logger.info("nsys collection started -> %s", report_path)
            return report_path
        last_err = err or out
        _check_perms(last_err)
        time.sleep(0.5 * (attempt + 1))
    raise NsysError(f"`nsys start` failed after retries: {last_err[-800:]}")


async def stop_collection() -> None:
    """Stop the current session's collection and flush its report to disk."""
    session = session_name()
    rc, out, err = await _run("nsys", "stop", f"--session={session}")
    if rc != 0:
        _check_perms(err or out)
        raise NsysError(f"`nsys stop` failed: {(err or out)[-800:]}")


def stop_collection_sync() -> None:
    """Blocking twin of stop_collection, for a `with nsys.range(...)` block in a non-async task body."""
    session = session_name()
    rc, out, err = _run_sync("nsys", "stop", f"--session={session}")
    if rc != 0:
        _check_perms(err or out)
        raise NsysError(f"`nsys stop` failed: {(err or out)[-800:]}")


async def run_stats(report_path: str, report_name: str) -> Optional[str]:
    """Run one `nsys stats` report over a .nsys-rep and return its CSV, or None on failure.

    Reports are version-dependent and any single one may be unsupported or empty; a failure
    here is expected for some report kinds and simply drops that section from the deck.
    """
    rc, out, err = await _run(
        "nsys", "stats", "--report", report_name, "--format", "csv", "--force-export=true", report_path
    )
    if rc != 0 or not out.strip():
        logger.debug("nsys stats %s produced no rows: %s", report_name, (err or out)[-300:])
        return None
    return out


def run_stats_sync(report_path: str, report_name: str) -> Optional[str]:
    """Blocking twin of run_stats, for the sync finalize path (a `with nsys.range(...)` block)."""
    rc, out, err = _run_sync(
        "nsys", "stats", "--report", report_name, "--format", "csv", "--force-export=true", report_path
    )
    if rc != 0 or not out.strip():
        logger.debug("nsys stats %s produced no rows: %s", report_name, (err or out)[-300:])
        return None
    return out
