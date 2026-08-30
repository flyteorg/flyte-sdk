"""Tests for dispatch/idempotency helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.clickup._dispatch import (
    DUPE_LABEL_KEY,
    DuplicateRun,
    blocking_run,
    launch_task,
    run_name_for,
)


def test_run_name_for_is_legal():
    name = run_name_for("abc123def456" * 10, prefix="cu")
    assert len(name) <= 30
    assert name.isalnum()
    assert name.startswith("cu")


def test_blocking_run_finds_live_run():
    live = MagicMock()
    live.phase = "RUNNING"
    with (
        patch("flyte.remote.Run.listall", return_value=iter([live])) as listall,
        patch("flyteplugins.clickup._dispatch._ensure_flyte_initialized"),
    ):
        assert blocking_run("k") is live
    listall.assert_called_once_with(with_labels={DUPE_LABEL_KEY: "k"}, limit=200)


def test_blocking_run_ignores_retriable():
    failed = MagicMock()
    failed.phase = "FAILED"
    with (
        patch("flyte.remote.Run.listall", return_value=iter([failed])),
        patch("flyteplugins.clickup._dispatch._ensure_flyte_initialized"),
    ):
        assert blocking_run("k") is None


def test_launch_task_raises_on_duplicate():
    live = MagicMock()
    live.phase = "RUNNING"
    live.name = "cux"
    live.url = "http://run"
    with (
        patch("flyteplugins.clickup._dispatch.blocking_run", return_value=live),
        patch("flyteplugins.clickup._dispatch._ensure_flyte_initialized"),
    ):
        with pytest.raises(DuplicateRun):
            launch_task(MagicMock(), key="k")


def test_launch_task_launches_with_labels():
    run = MagicMock()
    runner = MagicMock()
    runner.run.return_value = run
    with (
        patch("flyteplugins.clickup._dispatch.blocking_run", return_value=None),
        patch("flyteplugins.clickup._dispatch._ensure_flyte_initialized"),
        patch("flyteplugins.clickup._dispatch._allocate_name", return_value="cuabc"),
        patch("flyte.with_runcontext", return_value=runner) as with_runcontext,
    ):
        task = MagicMock()
        result = launch_task(task, key="k", repo="octo/repo", number=1)
    assert result is run
    with_runcontext.assert_called_once_with(name="cuabc", labels={DUPE_LABEL_KEY: "k"})
    runner.run.assert_called_once_with(task, repo="octo/repo", number=1)


def test_allocate_name_skips_existing():
    from flyteplugins.clickup._dispatch import _allocate_name

    with patch("flyteplugins.clickup._dispatch._run_exists", side_effect=[True, False]):
        name = _allocate_name("cuabc")
    assert name == "cuabc1"
