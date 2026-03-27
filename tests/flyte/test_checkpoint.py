"""Tests for flyte._checkpoint.AsyncCheckpoint."""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from flyte._checkpoint import AsyncCheckpoint, task_checkpoint_cache_key
from flyte.models import ActionID, Checkpoints, RawDataPath, TaskContext
from flyte.report import Report


def test_async_checkpoint_prev_exists() -> None:
    cp = AsyncCheckpoint("s3://bucket/out", None)
    assert not cp.prev_exists()
    cp2 = AsyncCheckpoint("s3://bucket/out", "s3://bucket/prev")
    assert cp2.prev_exists()


def test_async_checkpoint_file_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        remote_out = base / "checkpoint_out"
        remote_prev = base / "checkpoint_prev"
        remote_out.mkdir(parents=True)
        remote_prev.mkdir(parents=True)
        (remote_prev / "state.json").write_text(json.dumps({"step": 3}), encoding="utf-8")

        out_uri = remote_out.as_uri()
        prev_uri = remote_prev.as_uri()

        cp = AsyncCheckpoint(out_uri, prev_uri)
        restored = cp.load()
        assert restored is not None
        state_file = next(cp.path.rglob("state.json"))
        assert json.loads(state_file.read_text(encoding="utf-8"))["step"] == 3

        state_file.write_text(json.dumps({"step": 4}), encoding="utf-8")
        cp.save()

        out2 = (base / "checkpoint_out_2").as_uri()
        cp2 = AsyncCheckpoint(out2, out_uri)
        assert cp2.load() is not None
        state2 = next(cp2.path.rglob("state.json"))
        assert json.loads(state2.read_text(encoding="utf-8"))["step"] == 4


@pytest.mark.asyncio
async def test_async_checkpoint_load_save_aio_on_task_loop() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        remote_out = base / "checkpoint_out"
        remote_prev = base / "checkpoint_prev"
        remote_out.mkdir(parents=True)
        remote_prev.mkdir(parents=True)
        (remote_prev / "state.json").write_text(json.dumps({"step": 7}), encoding="utf-8")
        cp = AsyncCheckpoint(remote_out.as_uri(), remote_prev.as_uri())
        restored = await cp.load.aio()
        assert restored is not None
        state_file = next(cp.path.rglob("state.json"))
        assert json.loads(state_file.read_text(encoding="utf-8"))["step"] == 7
        await cp.save.aio(local_path=state_file)


def test_task_context_checkpoint_property_cached() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        out = (base / "out").as_uri()
        prev = (base / "prev").as_uri()
        tctx = TaskContext(
            action=ActionID(name="a0"),
            version="v1",
            raw_data_path=RawDataPath(path=str(base)),
            output_path=str(base / "outputs"),
            run_base_dir=str(base),
            report=Report(name="t"),
            checkpoints=Checkpoints(checkpoint_path=out, prev_checkpoint_path=prev),
        )
        k = task_checkpoint_cache_key()
        assert tctx.checkpoint is not None
        assert tctx.checkpoint is tctx.checkpoint
        assert k in tctx.data
        assert isinstance(tctx.data[k], AsyncCheckpoint)


def test_task_context_checkpoint_none_without_prefix() -> None:
    tctx = TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/rd"),
        output_path="/tmp/o",
        run_base_dir="/tmp",
        report=Report(name="t"),
        checkpoints=None,
    )
    assert tctx.checkpoint is None


def test_task_context_checkpoint_none_empty_dest() -> None:
    tctx = TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/rd"),
        output_path="/tmp/o",
        run_base_dir="/tmp",
        report=Report(name="t"),
        checkpoints=Checkpoints(checkpoint_path="  ", prev_checkpoint_path=None),
    )
    assert tctx.checkpoint is None
