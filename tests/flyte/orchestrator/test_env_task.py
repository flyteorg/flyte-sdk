"""Tests for `env.orchestrator.task`, which runs in the environment's image."""

import pytest

import flyte
from flyte.durable import sleep as durable_sleep
from flyte.sandbox._code_task import CodeTaskTemplate
from flyte.sandbox._task import SandboxedTaskTemplate


class TestEnvironmentOrchestratorTask:
    def test_bare_decorator(self):
        env = flyte.TaskEnvironment(name="orch-env", image="my-custom-image:latest")

        @env.orchestrator.task
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, SandboxedTaskTemplate)
        assert add.name == "orch-env.add"
        assert "orch-env.add" in env.tasks
        assert add.parent_env() is env
        assert add.image.base_image == "my-custom-image:latest"
        assert add.forward(1, 2) == 3

    def test_decorator_with_args(self):
        env = flyte.TaskEnvironment(name="orch-env2")

        @env.orchestrator.task(timeout_ms=5_000, retries=2, name="custom")
        def multiply(x: int, y: int) -> int:
            return x * y

        assert isinstance(multiply, SandboxedTaskTemplate)
        assert multiply.plugin_config.timeout_ms == 5_000
        assert multiply.retries.count == 2
        assert "custom" in env.tasks

    def test_may_call_durable_operations(self):
        """Unlike `flyte.orchestrator.task`, the env-bound variant allows durable operations."""
        env = flyte.TaskEnvironment(name="orch-env3")

        @env.orchestrator.task
        def waits(x: int) -> int:
            durable_sleep(1)
            return x

        assert isinstance(waits, SandboxedTaskTemplate)

    def test_code_string_mode(self):
        env = flyte.TaskEnvironment(name="orch-env4")

        @env.orchestrator.task
        def add(x: int, y: int) -> int:
            return x + y

        t = env.orchestrator.task("add(x, y)", inputs={"x": int, "y": int}, output=int, tasks=[add])
        assert isinstance(t, CodeTaskTemplate)
        assert t.func.__module__ == __name__

    def test_rejects_other_types(self):
        env = flyte.TaskEnvironment(name="orch-env5")
        with pytest.raises(TypeError, match=r"^task\(\) expects"):
            env.orchestrator.task(42)  # type: ignore[call-overload]

    def test_same_as_sandbox_orchestrator(self):
        env = flyte.TaskEnvironment(name="orch-env6")

        @env.sandbox.orchestrator
        def a(x: int) -> int:
            return x

        @env.orchestrator.task
        def b(x: int) -> int:
            return x

        assert type(a) is type(b)
        assert a.task_type == b.task_type
