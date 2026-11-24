"""A reusable worker pod's workload credential names the action each request is for, so its requests must name the
real running task and carry the ActionsService metadata."""

from types import SimpleNamespace

import pytest

from flyte._internal.controllers.remote._service_protocol import use_actions_service


@pytest.mark.parametrize(
    ("use_actions", "workload", "expected"),
    [
        (None, None, False),
        ("1", None, True),
        (None, "pod/acme/p/development/c1/ns/worker-0", True),
        (None, "acme/p/development/r1/a0/1", False),
        (None, "app/acme/p/development/web", False),
    ],
)
def test_reusable_pod_workloads_use_the_actions_service(monkeypatch, use_actions, workload, expected):
    for name, value in (("_U_USE_ACTIONS", use_actions), ("_U_WORKLOAD", workload)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert use_actions_service() is expected


def test_rust_controller_nests_under_the_running_task_inside_a_trace():
    pytest.importorskip("flyte_controller_base")
    from flyte._internal.controllers.remote._r_controller import _running_task

    task, trace = SimpleNamespace(name="real-task"), SimpleNamespace(name="outer-trace")
    assert _running_task(SimpleNamespace(task_action=task, action=trace)) is task
    assert _running_task(SimpleNamespace(task_action=None, action=task)) is task
