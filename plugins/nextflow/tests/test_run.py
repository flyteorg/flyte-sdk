"""Tests for flyteplugins.nextflow, using a fake `nextflow` executable on PATH."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flyteplugins.nextflow import NextflowError, run_nextflow
from flyteplugins.nextflow._image import _plugin_version_from_zip, nextflow_image
from flyteplugins.nextflow._run import (
    build_command,
    flyte_config,
    nextflow_env,
    nextflow_run_name,
    session_id_for,
    stable_raw_data_prefix,
)

FAKE_NEXTFLOW = """#!/usr/bin/env python3
import json, os, signal, sys, time
args = sys.argv[1:]
record = {
    "args": args,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("NF_FLYTE_", "NXF_"))},
    "configs": {},
    "params": None,
}
for i, a in enumerate(args):
    if a == "-c":
        record["configs"][args[i + 1]] = open(args[i + 1]).read()
    if a == "-params-file":
        record["params"] = json.load(open(args[i + 1]))
    if a in ("-with-report", "-with-timeline", "-with-dag"):
        open(args[i + 1], "w").write(f"<html><body>{a} <b>&</b></body></html>")
json.dump(record, open(os.environ["FAKE_NF_RECORD"], "w"))
print("N E X T F L O W  ~  fake", flush=True)
if os.environ.get("FAKE_NF_LONG_LINE"):
    print("x" * int(os.environ["FAKE_NF_LONG_LINE"]), flush=True)
open(".nextflow.log", "w").write("log line 1\\nlog line 2\\n")
if os.environ.get("FAKE_NF_WAIT"):
    def on_term(*_):
        print("got-term", flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    open(os.environ["FAKE_NF_STARTED"], "w").write("started")
    time.sleep(60)
sys.exit(int(os.environ.get("FAKE_NF_EXIT", "0")))
"""

RAW = "s3://bucket-raw/ob/org/proj/dev/r123/a0"


@pytest.fixture
def fake_nextflow(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "nextflow"
    exe.write_text(FAKE_NEXTFLOW)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.json"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NF_RECORD", str(record))
    monkeypatch.setenv("FAKE_NF_STARTED", str(tmp_path / "started"))
    monkeypatch.setenv("NF_FLYTE_VERSION", "0.1.0")
    return lambda: json.loads(record.read_text())


def make_ctx(attempt: int = 0, remote: bool = True, raw: str | None = None, trace: bool = False):
    task_action = SimpleNamespace(run_name="r123", name="a0")
    # inside @flyte.trace, `action` is the trace's pseudo-action; `task_action` is the real task
    action = SimpleNamespace(run_name="r123", name="trace-xyz") if trace else task_action
    return SimpleNamespace(
        raw_data_path=SimpleNamespace(path=raw or f"{RAW}/r123-a0-{attempt}"),
        run_base_dir="s3://bucket/metadata/org/proj/dev/r123",
        action=action,
        task_action=task_action,
        attempt_number=attempt,
        is_in_cluster=lambda: remote,
    )


@pytest.fixture
def task_ctx():
    ctx = make_ctx()
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=ctx):
        yield ctx


def arg(args, flag):
    return args[args.index(flag) + 1]


@pytest.mark.asyncio
async def test_run_nextflow_inside_a_task(fake_nextflow, task_ctx):
    result = await run_nextflow(
        "nf-core/demo",
        revision="1.2.0",
        profile="test",
        params={"input": "s3://bucket/samples.csv", "skip_qc": True},
        outdir="results",
    )

    rec = fake_nextflow()
    args = rec["args"]
    assert arg(args, "run") == "nf-core/demo"
    # work and output dirs live under the task's raw data prefix, shared by all attempts
    assert arg(args, "-w") == f"{RAW}/nextflow-work"
    assert arg(args, "--outdir") == f"{RAW}/results"
    assert arg(args, "-r") == "1.2.0"
    assert arg(args, "-profile") == "test"
    assert arg(args, "-name") == "flyte-r123-a0-0"
    assert "-resume" not in args
    assert rec["params"] == {"input": "s3://bucket/samples.csv", "skip_qc": True}

    # the generated config routes every process to nf-flyte, and it comes before `run`
    assert args.index("-c") < args.index("run")
    (generated,) = rec["configs"].values()
    assert "id 'nf-flyte@0.1.0'" in generated
    assert "process.executor = 'flyte'" in generated

    env = rec["env"]
    assert env["NF_FLYTE_RUN_NAME"] == "r123"
    assert env["NF_FLYTE_PARENT_ACTION"] == "a0"
    assert env["NXF_ANSI_LOG"] == "false"
    # resume state lives in the work dir, not the throwaway launch dir
    assert env["NXF_CLOUDCACHE_PATH"] == f"{RAW}/nextflow-work/.nextflow-cache"
    assert env["NXF_UUID"] == session_id_for(f"{RAW}/nextflow-work")
    assert env["NXF_IGNORE_RESUME_HISTORY"] == "true"

    assert result is not None
    assert result.path == f"{RAW}/results"


@pytest.mark.asyncio
async def test_retried_task_resumes_the_same_session(fake_nextflow):
    runs = []
    for attempt in (0, 1):
        with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=make_ctx(attempt=attempt)):
            await run_nextflow("nf-core/demo")
        runs.append(fake_nextflow())

    first, retry = runs
    assert arg(first["args"], "-w") == arg(retry["args"], "-w") == f"{RAW}/nextflow-work"
    assert "-resume" not in first["args"]
    assert arg(retry["args"], "-resume") == first["env"]["NXF_UUID"] == retry["env"]["NXF_UUID"]


@pytest.mark.asyncio
async def test_resume_across_runs_with_a_fixed_work_dir(fake_nextflow, task_ctx):
    await run_nextflow("main.nf", work_dir="s3://other/work", resume=True, extra_args=["-stub"])
    args = fake_nextflow()["args"]
    assert arg(args, "-w") == "s3://other/work"
    assert arg(args, "-resume") == session_id_for("s3://other/work") == session_id_for("s3://other/work/")
    assert "-stub" in args
    assert "--outdir" not in args and "-params-file" not in args


@pytest.mark.asyncio
async def test_parents_task_actions_under_the_task_not_a_trace(fake_nextflow):
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=make_ctx(trace=True)):
        await run_nextflow("main.nf")
    assert fake_nextflow()["env"]["NF_FLYTE_PARENT_ACTION"] == "a0"


@pytest.mark.asyncio
async def test_local_mode_uses_nextflows_own_executors(fake_nextflow, tmp_path):
    ctx = make_ctx(remote=False, raw=str(tmp_path / "raw"))
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=ctx):
        result = await run_nextflow("main.nf", profile="docker", outdir="results")
    rec = fake_nextflow()
    assert rec["configs"] == {}  # no nf-flyte
    assert "NF_FLYTE_RUN_NAME" not in rec["env"]
    assert rec["env"]["NXF_CACHE_DIR"] == str(tmp_path / "raw" / "nextflow-work" / ".nextflow-cache")
    assert "NXF_CLOUDCACHE_PATH" not in rec["env"]
    assert arg(rec["args"], "-w") == str(tmp_path / "raw" / "nextflow-work")
    assert arg(rec["args"], "-profile") == "docker"
    assert result.path == str(tmp_path / "raw" / "results")


@pytest.mark.asyncio
async def test_run_nextflow_with_extra_config_text(fake_nextflow, task_ctx):
    await run_nextflow("main.nf", config="process.cpus = 4\n")
    configs = fake_nextflow()["configs"]
    assert len(configs) == 2
    assert "process.cpus = 4\n" in configs.values()


@pytest.mark.asyncio
async def test_run_nextflow_raises_with_log_tail(fake_nextflow, task_ctx, monkeypatch):
    monkeypatch.setenv("FAKE_NF_EXIT", "3")
    with pytest.raises(NextflowError) as e:
        await run_nextflow("main.nf")
    assert e.value.exit_code == 3
    assert "N E X T F L O W  ~  fake" in str(e.value)
    assert "log line 2" in str(e.value)


@pytest.mark.asyncio
async def test_output_lines_longer_than_64k(fake_nextflow, task_ctx, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_NF_LONG_LINE", str(200_000))
    await run_nextflow("main.nf")
    assert "x" * 200_000 in capsys.readouterr().out


async def _wait_started(tmp_path: Path):
    for _ in range(100):
        if (tmp_path / "started").exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("fake nextflow did not start")


@pytest.mark.asyncio
async def test_sigterm_is_forwarded_to_nextflow(fake_nextflow, task_ctx, monkeypatch, tmp_path):
    # the pod gets SIGTERM on abort; Nextflow must get it too, to shut down cleanly
    monkeypatch.setenv("FAKE_NF_WAIT", "1")
    run = asyncio.create_task(run_nextflow("main.nf"))
    await _wait_started(tmp_path)
    os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(NextflowError) as e:
        await asyncio.wait_for(run, timeout=10)
    assert e.value.exit_code == 143
    assert "got-term" in str(e.value)
    # the handler is only installed while Nextflow runs
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


@pytest.mark.asyncio
async def test_cancellation_terminates_nextflow(fake_nextflow, task_ctx, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FAKE_NF_WAIT", "1")
    run = asyncio.create_task(run_nextflow("main.nf"))
    await _wait_started(tmp_path)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=10)
    assert "got-term" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_reports_are_rendered_into_the_flyte_report(fake_nextflow, task_ctx):
    tabs, flushed = {}, []

    class Tab:
        def __init__(self, name):
            self.name = name

        def replace(self, content):
            tabs[self.name] = content

    async def flush():
        flushed.append(True)

    with (
        patch("flyte.report.get_tab", side_effect=Tab),
        patch("flyte.report.flush", SimpleNamespace(aio=flush)),
    ):
        await run_nextflow("main.nf", report=True)

    args = fake_nextflow()["args"]
    assert {"-with-report", "-with-timeline", "-with-dag"} <= set(args)
    assert set(tabs) == {"Nextflow report", "Timeline", "DAG"}
    # full HTML documents, escaped into an iframe
    assert tabs["Nextflow report"].startswith('<iframe srcdoc="&lt;html&gt;')
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in tabs["Nextflow report"]
    assert flushed == [True]


@pytest.mark.asyncio
async def test_run_nextflow_needs_a_task_and_an_object_store_on_a_cluster(fake_nextflow):
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=None):
        with pytest.raises(RuntimeError, match="inside a Flyte task"):
            await run_nextflow("main.nf")
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=make_ctx(raw="/local/raw")):
        with pytest.raises(ValueError, match="object store work directory"):
            await run_nextflow("main.nf")
    # any object store works: nf-flyte stages task data through Flyte
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=make_ctx(raw="gs://bucket/raw")):
        await run_nextflow("main.nf")


def test_nextflow_run_name():
    assert nextflow_run_name("ur9npl2kmzqzqpwgrgsg", "a0", 1) == "flyte-ur9npl2kmzqzqpwgrgsg-a0-1"
    # made valid for Nextflow: letters, digits, - and _, starting with a letter, <= 80 chars
    name = nextflow_run_name("Run.With/Odd__Chars", "x" * 100, 0)
    assert re.fullmatch(r"[a-z](?:[a-z\d]|[-_](?=[a-z\d])){0,79}", name), name


@pytest.mark.asyncio
async def test_caller_can_name_the_run(fake_nextflow, task_ctx):
    await run_nextflow("main.nf", extra_args=["-name", "mine"])
    args = fake_nextflow()["args"]
    assert args.count("-name") == 1 and arg(args, "-name") == "mine"


def test_stable_raw_data_prefix():
    assert stable_raw_data_prefix(f"{RAW}/r123-a0-0", "r123", "a0", 0) == RAW
    assert stable_raw_data_prefix(f"{RAW}/r123-a0-2/", "r123", "a0", 2) == RAW
    # anything else is left alone
    assert stable_raw_data_prefix("/tmp/local/raw", "r123", "a0", 0) == "/tmp/local/raw"
    assert stable_raw_data_prefix(f"{RAW}/r123-a0-1", "r123", "a0", 0) == f"{RAW}/r123-a0-1"


def test_build_command_minimal():
    assert build_command("main.nf", work_dir="s3://b/w", configs=[Path("/x/flyte.config")]) == [
        "nextflow",
        "-c",
        "/x/flyte.config",
        "run",
        "main.nf",
        "-w",
        "s3://b/w",
    ]


def test_flyte_config_and_env():
    assert "id 'nf-flyte@9.9.9'" in flyte_config("9.9.9")
    env = nextflow_env(
        {"PATH": "/bin", "NXF_ANSI_LOG": "true"},
        session_id="sid",
        cache_path="s3://b/w/.nextflow-cache",
        run_name="r1",
        action_name="a3",
    )
    assert env == {
        "PATH": "/bin",
        "NXF_ANSI_LOG": "true",
        "NXF_UUID": "sid",
        "NXF_CLOUDCACHE_PATH": "s3://b/w/.nextflow-cache",
        "NXF_IGNORE_RESUME_HISTORY": "true",
        "NF_FLYTE_RUN_NAME": "r1",
        "NF_FLYTE_PARENT_ACTION": "a3",
    }


def test_plugin_version_from_zip(tmp_path):
    z = tmp_path / "nf-flyte-0.2.0.zip"
    z.write_bytes(b"")
    assert _plugin_version_from_zip(z) == "0.2.0"
    with pytest.raises(ValueError):
        _plugin_version_from_zip(tmp_path / "plugin.zip")
    # must not require the zip: the image definition is evaluated again inside the task pod
    assert _plugin_version_from_zip(tmp_path / "missing" / "nf-flyte-1.0.0.zip") == "1.0.0"


def test_nextflow_image_from_registry_and_local_zip(tmp_path):
    registry = nextflow_image(nextflow_version="25.10.0", nf_flyte="0.1.0")
    z = tmp_path / "nf-flyte-0.2.0.zip"
    z.write_bytes(b"")
    local = nextflow_image(nf_flyte=z, name="nf-dev")
    # different plugin sources must produce different images
    assert registry != local
