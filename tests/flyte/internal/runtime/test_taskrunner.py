"""Tests for taskrunner behavior.

`run_task`: the `controller` argument is optional. Clustered/jobset tasks run with no controller
(they never enqueue subtasks); the only controller touchpoint on the leaf path is
`finalize_parent_action`, which must be skipped when there is none.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flyte._internal.runtime import taskrunner
from flyte._internal.runtime.taskrunner import extract_download_run_upload, run_task


class _FakeTask:
    async def execute(self, **kwargs):
        return {"out": 1}


def test_run_task_without_controller_skips_finalize():
    tctx = SimpleNamespace(action="act-1")
    out, err = asyncio.run(run_task(tctx=tctx, controller=None, task=_FakeTask(), inputs={}))
    assert err is None
    assert out == {"out": 1}


def test_run_task_with_controller_finalizes():
    tctx = SimpleNamespace(action="act-1")
    controller = AsyncMock()
    _out, err = asyncio.run(run_task(tctx=tctx, controller=controller, task=_FakeTask(), inputs={}))
    assert err is None
    controller.finalize_parent_action.assert_awaited_once_with("act-1")


# --- clustered worker non-zero exit on failure ---


def _run_extract():
    return asyncio.run(
        extract_download_run_upload(
            SimpleNamespace(name="t"),
            action=SimpleNamespace(name="a0"),
            controller=None,
            raw_data_path=SimpleNamespace(path_rewrite=None),
            output_path="s3://bucket/outputs",
            run_base_dir="s3://bucket",
            version="v1",
        )
    )


def _patch_failure(monkeypatch):
    err = SimpleNamespace(err=SimpleNamespace(), recoverable=True)
    monkeypatch.setattr(taskrunner, "convert_and_run", AsyncMock(return_value=(None, err)))
    monkeypatch.setattr(taskrunner, "upload_error", AsyncMock(return_value="s3://bucket/outputs/error.pb"))


def test_clustered_worker_exits_nonzero_on_failure(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-123")
    _patch_failure(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        _run_extract()
    assert exc_info.value.code == taskrunner._CLUSTERED_FAILURE_EXIT_CODE


def test_non_clustered_failure_does_not_exit(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    upload_error = AsyncMock(return_value="s3://bucket/outputs/error.pb")
    err = SimpleNamespace(err=SimpleNamespace(), recoverable=True)
    monkeypatch.setattr(taskrunner, "convert_and_run", AsyncMock(return_value=(None, err)))
    monkeypatch.setattr(taskrunner, "upload_error", upload_error)

    # Returns normally (process would exit 0); the backend reads error.pb.
    assert _run_extract() is None
    upload_error.assert_awaited_once()


def test_clustered_worker_success_does_not_exit(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-123")
    monkeypatch.setattr(taskrunner, "convert_and_run", AsyncMock(return_value=(SimpleNamespace(), None)))
    monkeypatch.setattr(taskrunner, "upload_outputs", AsyncMock())

    assert _run_extract() is None
