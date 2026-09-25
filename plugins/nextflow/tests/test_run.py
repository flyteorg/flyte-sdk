"""Tests for flyteplugins.nextflow, using a fake `nextflow` executable on PATH."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flyteplugins.nextflow import NextflowError, run_nextflow
from flyteplugins.nextflow._image import _plugin_version_from_zip, nextflow_image
from flyteplugins.nextflow._run import build_command, flyte_config, nextflow_env

FAKE_NEXTFLOW = """#!/usr/bin/env python3
import json, os, sys
record = {
    "args": sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("NF_FLYTE_", "NXF_"))},
    "configs": {},
    "params": None,
}
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "-c":
        record["configs"][args[i + 1]] = open(args[i + 1]).read()
    if a == "-params-file":
        record["params"] = json.load(open(args[i + 1]))
json.dump(record, open(os.environ["FAKE_NF_RECORD"], "w"))
print("N E X T F L O W  ~  fake")
open(".nextflow.log", "w").write("log line 1\\nlog line 2\\n")
sys.exit(int(os.environ.get("FAKE_NF_EXIT", "0")))
"""


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
    monkeypatch.setenv("NF_FLYTE_VERSION", "0.1.0")
    return lambda: json.loads(record.read_text())


@pytest.fixture
def task_ctx():
    ctx = SimpleNamespace(
        run_base_dir="s3://bucket/metadata/org/proj/dev/r123",
        action=SimpleNamespace(run_name="r123", name="a0"),
    )
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=ctx):
        yield ctx


@pytest.mark.asyncio
async def test_run_nextflow_inside_a_task(fake_nextflow, task_ctx):
    result = await run_nextflow(
        "nf-core/demo",
        revision="1.0.2",
        profile="test",
        params={"input": "s3://bucket/samples.csv", "skip_qc": True},
        outdir="results",
    )

    rec = fake_nextflow()
    args = rec["args"]
    assert args[args.index("run") + 1] == "nf-core/demo"
    assert args[args.index("-w") + 1] == "s3://bucket/metadata/org/proj/dev/r123/nextflow-work"
    assert args[args.index("-r") + 1] == "1.0.2"
    assert args[args.index("-profile") + 1] == "test"
    assert args[args.index("--outdir") + 1] == "s3://bucket/metadata/org/proj/dev/r123/results"
    assert rec["params"] == {"input": "s3://bucket/samples.csv", "skip_qc": True}

    # the generated config routes every process to nf-flyte, and it comes before `run`
    assert args.index("-c") < args.index("run")
    (generated,) = rec["configs"].values()
    assert "id 'nf-flyte@0.1.0'" in generated
    assert "process.executor = 'flyte'" in generated

    # nf-flyte learns which run and parent action to add task actions to from the env
    assert rec["env"]["NF_FLYTE_RUN_NAME"] == "r123"
    assert rec["env"]["NF_FLYTE_PARENT_ACTION"] == "a0"
    assert rec["env"]["NXF_ANSI_LOG"] == "false"

    assert result is not None
    assert result.path == "s3://bucket/metadata/org/proj/dev/r123/results"


@pytest.mark.asyncio
async def test_run_nextflow_without_outdir_returns_none(fake_nextflow, task_ctx):
    assert await run_nextflow("main.nf", work_dir="s3://other/work", resume=True, extra_args=["-stub"]) is None
    args = fake_nextflow()["args"]
    assert args[args.index("-w") + 1] == "s3://other/work"
    assert "-resume" in args and "-stub" in args
    assert "--outdir" not in args and "-params-file" not in args


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
async def test_run_nextflow_needs_a_task_and_s3(fake_nextflow):
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=None):
        with pytest.raises(RuntimeError, match="inside a Flyte task"):
            await run_nextflow("main.nf")
    ctx = SimpleNamespace(run_base_dir="/tmp/local", action=SimpleNamespace(run_name="r", name="a0"))
    with patch("flyteplugins.nextflow._run.flyte.ctx", return_value=ctx):
        with pytest.raises(ValueError, match="S3 work directory"):
            await run_nextflow("main.nf")


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
    env = nextflow_env({"PATH": "/bin", "NXF_ANSI_LOG": "true"}, run_name="r1", action_name="a3")
    assert env == {"PATH": "/bin", "NXF_ANSI_LOG": "true", "NF_FLYTE_RUN_NAME": "r1", "NF_FLYTE_PARENT_ACTION": "a3"}


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
