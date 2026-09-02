"""
Tests for `flyte.control_plane_available()`.

The predicate answers "can this process submit work to a control plane" — inside a task it is
decided by how the task is orchestrated (mode), outside a task by whether a client is
configured. See the function's docstring for the full decision table.
"""

import pathlib

import pytest

import flyte
from flyte._context import internal_ctx
from flyte._initialize import _InitConfig
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report


def _tctx(mode: str) -> TaskContext:
    return TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/rd"),
        output_path="/tmp/o",
        run_base_dir="/tmp",
        report=Report(name="t"),
        mode=mode,
    )


def test_false_when_uninitialized(monkeypatch):
    monkeypatch.setattr("flyte._initialize._init_config", None)
    assert flyte.control_plane_available() is False


def test_false_when_initialized_without_client(monkeypatch):
    monkeypatch.setattr("flyte._initialize._init_config", _InitConfig(root_dir=pathlib.Path.cwd()))
    assert flyte.control_plane_available() is False


def test_true_when_client_configured(monkeypatch):
    monkeypatch.setattr(
        "flyte._initialize._init_config",
        _InitConfig(root_dir=pathlib.Path.cwd(), client=object()),
    )
    assert flyte.control_plane_available() is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("remote", True), ("hybrid", True), ("local", False)],
)
def test_task_context_mode_decides(monkeypatch, mode, expected):
    # Inside a task, the orchestration mode wins over client presence: `flyte run --local`
    # configures a client too, but a locally-orchestrated run has no control plane behind it.
    monkeypatch.setattr(
        "flyte._initialize._init_config",
        _InitConfig(root_dir=pathlib.Path.cwd(), client=object()),
    )
    with internal_ctx().replace_task_context(_tctx(mode)):
        assert flyte.control_plane_available() is expected
