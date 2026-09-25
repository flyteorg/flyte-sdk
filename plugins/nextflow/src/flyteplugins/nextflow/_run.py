from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from flyteplugins.nextflow._image import DEFAULT_NF_FLYTE_VERSION

import flyte
from flyte.io import Dir
from flyte.storage import join

_LOG_TAIL_LINES = 80


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
    extra_args: Sequence[str] = (),
) -> Dir | None:
    """
    Run a Nextflow pipeline from inside a Flyte task, with every Nextflow task running as a
    child action of the calling task. Must be called from a task using `nextflow_image()`.

    :param pipeline: Pipeline to run: a repository (e.g. `nf-core/rnaseq`), URL or local path.
    :param revision: Pipeline revision (git tag, branch or commit).
    :param profile: Comma separated Nextflow config profiles, e.g. `test`.
    :param params: Pipeline parameters, passed through a params file.
    :param outdir: Output directory, passed as `--outdir`. A relative path is placed under this
        run's storage. The directory is returned as a `Dir`.
    :param work_dir: S3 work directory. Defaults to a `nextflow-work` prefix in this run's
        storage; set a fixed location to use `resume` across runs.
    :param config: Extra Nextflow config: a path to a config file, or config text.
    :param resume: Pass `-resume`.
    :param extra_args: Extra arguments for `nextflow run`.
    :return: The `outdir` as a `Dir`, or None when no `outdir` is given.
    """
    tctx = flyte.ctx()
    if tctx is None:
        raise RuntimeError("run_nextflow() must be called from inside a Flyte task")

    work_dir = work_dir or join(tctx.run_base_dir, "nextflow-work")
    if outdir is not None and "://" not in outdir:
        outdir = join(tctx.run_base_dir, outdir)
    if not work_dir.startswith("s3://"):
        raise ValueError(f"The nf-flyte executor needs an S3 work directory, got '{work_dir}'")

    with tempfile.TemporaryDirectory(prefix="nextflow-") as launch_dir:
        launch = Path(launch_dir)
        (launch / "flyte.config").write_text(flyte_config(os.environ.get("NF_FLYTE_VERSION", DEFAULT_NF_FLYTE_VERSION)))
        configs = [launch / "flyte.config"]
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
            resume=resume,
            extra_args=extra_args,
        )
        env = nextflow_env(os.environ, run_name=tctx.action.run_name, action_name=tctx.action.name)
        exit_code, output_tail = await _stream(cmd, cwd=launch, env=env)
        if exit_code != 0:
            raise NextflowError(exit_code, output_tail, _tail(launch / ".nextflow.log"))

    return Dir.from_existing_remote(outdir) if outdir is not None else None


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
    resume: bool = False,
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
    if resume:
        cmd.append("-resume")
    cmd += list(extra_args)
    if outdir is not None:
        cmd += ["--outdir", outdir]
    return cmd


def nextflow_env(base: Mapping[str, str], *, run_name: str, action_name: str) -> dict[str, str]:
    """Environment for the Nextflow head: nf-flyte reads the run it adds task actions to from here."""
    env = dict(base)
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
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            print(line, flush=True)
            tail.append(line)
        return await proc.wait(), list(tail)
    except asyncio.CancelledError:
        # the Flyte task was aborted: let Nextflow abort its task actions before exiting
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
        raise


def _tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:]
