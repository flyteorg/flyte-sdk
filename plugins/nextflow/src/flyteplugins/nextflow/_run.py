from __future__ import annotations

import asyncio
import html
import json
import os
import re
import signal
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from flyteplugins.nextflow._image import DEFAULT_NF_FLYTE_VERSION

import flyte
from flyte.io import Dir
from flyte.storage import join

_LOG_TAIL_LINES = 80
_READ_CHUNK = 64 * 1024
# Kubernetes gives a terminating pod 30s by default before SIGKILL
_SHUTDOWN_GRACE_SECONDS = 20

# Nextflow report files rendered into the task's Flyte report, by tab name
_REPORTS = {"Nextflow report": "report.html", "Timeline": "timeline.html", "DAG": "dag.html"}


class NextflowError(RuntimeError):
    """Raised when `nextflow run` exits with a non-zero status."""

    def __init__(self, exit_code: int, output_tail: Sequence[str], log_tail: Sequence[str]):
        self.exit_code = exit_code
        message = [f"nextflow exited with status {exit_code}", *output_tail]
        if log_tail:
            message += ["--- .nextflow.log (tail) ---", *log_tail]
        super().__init__("\n".join(message))


async def run_nextflow(
    pipeline: str,
    *,
    revision: str | None = None,
    profile: str | None = None,
    params: Mapping[str, Any] | None = None,
    outdir: str | None = None,
    work_dir: str | None = None,
    config: str | Path | None = None,
    resume: bool = False,
    report: bool = False,
    extra_args: Sequence[str] = (),
) -> Dir | None:
    """
    Run a Nextflow pipeline from inside a Flyte task. On a Flyte cluster, every Nextflow task
    runs as a child action of the calling task, through the nf-flyte executor; the calling task
    must use `nextflow_image()`. In a local run, Nextflow uses its own executors instead
    (`local` unless `config` or `profile` selects another, e.g. `docker`).

    :param pipeline: Pipeline to run: a repository (e.g. `nf-core/rnaseq`), URL or local path.
    :param revision: Pipeline revision (git tag, branch or commit).
    :param profile: Comma separated Nextflow config profiles, e.g. `test`.
    :param params: Pipeline parameters, passed through a params file.
    :param outdir: Output directory, passed as `--outdir`. A relative path is placed under this
        task's raw data prefix. The directory is returned as a `Dir`.
    :param work_dir: Work directory; on a cluster, an object store location Flyte can read.
        Defaults to a `nextflow-work` prefix under this task's raw data prefix, which is shared
        by all attempts of the task, so a retried task resumes where the failed attempt stopped.
        Set a fixed location to resume across runs.
    :param config: Extra Nextflow config: a path to a config file, or config text.
    :param resume: Resume a previous run that used the same `work_dir`. Always on when the task
        is being retried.
    :param report: Render Nextflow's execution report, timeline and DAG into the task's Flyte
        report. The task must be declared with `report=True`.
    :param extra_args: Extra arguments for `nextflow run`.
    :return: The `outdir` as a `Dir`, or None when no `outdir` is given.
    """
    tctx = flyte.ctx()
    if tctx is None:
        raise RuntimeError("run_nextflow() must be called from inside a Flyte task")
    remote = tctx.is_in_cluster()

    storage_root = stable_raw_data_prefix(
        tctx.raw_data_path.path, tctx.task_action.run_name, tctx.task_action.name, tctx.attempt_number
    )
    work_dir = work_dir or join(storage_root, "nextflow-work")
    if outdir is not None and "://" not in outdir and not os.path.isabs(outdir):
        outdir = join(storage_root, outdir)
    if remote and "://" not in work_dir:
        raise ValueError(f"The nf-flyte executor needs an object store work directory Flyte can read, got '{work_dir}'")

    session_id = session_id_for(work_dir)
    resume = resume or tctx.attempt_number > 0

    with tempfile.TemporaryDirectory(prefix="nextflow-") as launch_dir:
        launch = Path(launch_dir)
        configs = []
        if remote:
            (launch / "flyte.config").write_text(
                flyte_config(os.environ.get("NF_FLYTE_VERSION", DEFAULT_NF_FLYTE_VERSION))
            )
            configs.append(launch / "flyte.config")
        if config is not None:
            configs.append(_user_config(config, launch))

        params_file = None
        if params:
            params_file = launch / "params.json"
            params_file.write_text(json.dumps(dict(params)))

        cmd = build_command(
            pipeline,
            work_dir=work_dir,
            configs=configs,
            revision=revision,
            profile=profile,
            params_file=params_file,
            outdir=outdir,
            resume_session=session_id if resume else None,
            run_name=None
            if "-name" in extra_args
            else nextflow_run_name(tctx.task_action.run_name, tctx.task_action.name, tctx.attempt_number),
            reports_dir=launch if report else None,
            extra_args=extra_args,
        )
        env = nextflow_env(
            os.environ,
            session_id=session_id,
            cache_path=join(work_dir, ".nextflow-cache"),
            run_name=tctx.task_action.run_name if remote else None,
            action_name=tctx.task_action.name if remote else None,
        )
        try:
            exit_code, output_tail = await _stream(cmd, cwd=launch, env=env)
        finally:
            if report:
                await _publish_reports(launch)
        if exit_code != 0:
            raise NextflowError(exit_code, output_tail, _tail(launch / ".nextflow.log"))

    return Dir.from_existing_remote(outdir) if outdir is not None else None


def stable_raw_data_prefix(raw_data_path: str, run_name: str, action_name: str, attempt: int) -> str:
    """
    The part of the task's raw data path shared by all of its attempts. Flyte gives each attempt
    its own raw data prefix ending in `<run>-<action>-<attempt>`; resuming after a retry needs the
    work dir to stay put, so drop that last segment.
    """
    path = raw_data_path.rstrip("/")
    parent, _, last = path.rpartition("/")
    if parent and last == f"{run_name}-{action_name}-{attempt}":
        return parent
    return path


def session_id_for(work_dir: str) -> str:
    """
    Nextflow session id for a work dir. Resuming needs the id of the previous session, which
    Nextflow normally looks up in `.nextflow/history` under the launch dir; that doesn't survive
    the task pod, so derive the id from the work dir instead.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, work_dir.rstrip("/")))


def nextflow_run_name(run_name: str, action_name: str, attempt: int) -> str:
    """
    Nextflow normally picks an unused run name from its history file; with history disabled
    (`NXF_IGNORE_RESUME_HISTORY`), the run name has to be given. It must start with a letter and
    contain only letters, digits, `-` and `_`, at most 80 characters.
    """
    name = re.sub(r"[^a-z0-9]+", "-", f"flyte-{run_name}-{action_name}-{attempt}".lower()).strip("-")
    return name[:80].rstrip("-")


def flyte_config(nf_flyte_version: str) -> str:
    """Nextflow config that routes every process to the nf-flyte executor."""
    return f"""\
plugins {{
    id 'nf-flyte@{nf_flyte_version}'
}}

process.executor = 'flyte'
"""


def build_command(
    pipeline: str,
    *,
    work_dir: str,
    configs: Sequence[Path],
    revision: str | None = None,
    profile: str | None = None,
    params_file: Path | None = None,
    outdir: str | None = None,
    resume_session: str | None = None,
    run_name: str | None = None,
    reports_dir: Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    cmd = ["nextflow"]
    for c in configs:
        cmd += ["-c", str(c)]
    cmd += ["run", pipeline, "-w", work_dir]
    if revision:
        cmd += ["-r", revision]
    if profile:
        cmd += ["-profile", profile]
    if params_file is not None:
        cmd += ["-params-file", str(params_file)]
    if run_name:
        cmd += ["-name", run_name]
    if resume_session:
        cmd += ["-resume", resume_session]
    if reports_dir is not None:
        cmd += [
            "-with-report", str(reports_dir / _REPORTS["Nextflow report"]),
            "-with-timeline", str(reports_dir / _REPORTS["Timeline"]),
            "-with-dag", str(reports_dir / _REPORTS["DAG"]),
        ]  # fmt: skip
    cmd += list(extra_args)
    if outdir is not None:
        cmd += ["--outdir", outdir]
    return cmd


def nextflow_env(
    base: Mapping[str, str],
    *,
    session_id: str,
    cache_path: str,
    run_name: str | None = None,
    action_name: str | None = None,
) -> dict[str, str]:
    """
    Environment for the Nextflow head. The session id and cloud cache keep resume state outside
    the launch dir; nf-flyte reads the run and parent action it adds task actions to.
    """
    env = dict(base)
    env["NXF_UUID"] = session_id
    if "://" in cache_path:
        env["NXF_CLOUDCACHE_PATH"] = cache_path
    else:
        # the cloud cache only takes object store paths; locally, move `.nextflow` instead
        env["NXF_CACHE_DIR"] = cache_path
    # resume by the explicit session id: don't require it in the (per-pod) history file
    env["NXF_IGNORE_RESUME_HISTORY"] = "true"
    if run_name and action_name:
        env["NF_FLYTE_RUN_NAME"] = run_name
        env["NF_FLYTE_PARENT_ACTION"] = action_name
    env.setdefault("NXF_ANSI_LOG", "false")
    return env


def _user_config(config: str | Path, launch: Path) -> Path:
    if isinstance(config, Path):
        return config
    if "\n" not in config and config.endswith(".config") and Path(config).is_file():
        return Path(config)
    path = launch / "user.config"
    path.write_text(config)
    return path


async def _stream(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> tuple[int, list[str]]:
    print("+", " ".join(cmd), flush=True)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=dict(env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    tail: deque[str] = deque(maxlen=_LOG_TAIL_LINES)
    uninstall = _forward_sigterm(proc)
    try:
        await _pump_lines(proc.stdout, tail)
        return await proc.wait(), list(tail)
    except asyncio.CancelledError:
        # keep relaying output while Nextflow shuts down: its abort messages belong in the logs
        await _terminate(proc, _pump_lines(proc.stdout, tail))
        raise
    finally:
        uninstall()


def _forward_sigterm(proc: asyncio.subprocess.Process):
    """
    Pass SIGTERM on to Nextflow so it can shut down cleanly (abort its task actions, save its
    cache) when the task pod is deleted, e.g. on abort. Without a handler, the Flyte runtime
    process dies on SIGTERM and never reaches the cancellation path.
    """

    def forward():
        if proc.returncode is None:
            proc.send_signal(signal.SIGTERM)

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, forward)
    except (NotImplementedError, RuntimeError, ValueError):
        # not the main thread's loop, or not supported on this platform
        return lambda: None
    return lambda: loop.remove_signal_handler(signal.SIGTERM)


async def _terminate(proc: asyncio.subprocess.Process, drain):
    if proc.returncode is None:
        proc.terminate()
    try:
        await asyncio.wait_for(asyncio.gather(drain, proc.wait()), timeout=_SHUTDOWN_GRACE_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()


async def _pump_lines(stream: asyncio.StreamReader | None, tail: deque[str]):
    """Echo output line by line. Reads in chunks: `readline()` fails on lines over 64 KiB."""
    assert stream is not None
    buf = b""
    while chunk := await stream.read(_READ_CHUNK):
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            _emit(raw, tail)
    if buf:
        _emit(buf, tail)


def _emit(raw: bytes, tail: deque[str]):
    line = raw.decode(errors="replace").rstrip("\r")
    print(line, flush=True)
    tail.append(line)


async def _publish_reports(launch: Path):
    import flyte.report

    published = False
    for tab, name in _REPORTS.items():
        path = launch / name
        if path.exists():
            # the reports are full HTML documents; a tab's content goes inside a <div>
            doc = html.escape(path.read_text(errors="replace"), quote=True)
            flyte.report.get_tab(tab).replace(
                f'<iframe srcdoc="{doc}" style="width:100%;height:85vh;border:0"></iframe>'
            )
            published = True
    if published:
        await flyte.report.flush.aio()


def _tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:]
