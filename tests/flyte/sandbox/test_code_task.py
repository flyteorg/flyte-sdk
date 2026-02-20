"""Tests for flyte.sandbox.orchestrate(), flyte.sandbox.orchestrate_local(),
and TaskEnvironment.sandbox.orchestrate().
"""

from __future__ import annotations

import pytest

from flyte.sandbox import orchestrate
from flyte.sandbox._code_task import (
    CodeTaskTemplate,
    _classify_refs,
)
from flyte.sandbox._config import SandboxedConfig
from flyte.sandbox._source import prepare_code_source
from flyte.sandbox._task import SandboxedTaskTemplate

# ---------------------------------------------------------------------------
# prepare_code_source
# ---------------------------------------------------------------------------


class TestPrepareCodeSource:
    def test_expression_becomes_result(self):
        result = prepare_code_source("x + y")
        assert "__result__" in result
        assert "x + y" in result

    def test_assignment_appends_result(self):
        result = prepare_code_source("result = x + y")
        # Should contain the assignment AND __result__ = result
        assert "result = x + y" in result
        assert "__result__ = result" in result
        assert result.endswith("__result__")

    def test_multiline_last_expression(self):
        src = """\
        a = x + 1
        b = a * 2
        b
        """
        result = prepare_code_source(src)
        assert "__result__" in result
        # The last line (expression `b`) becomes __result__ = b
        assert "b" in result

    def test_multiline_last_assignment(self):
        src = """\
        partial = x + y
        result = partial * 2
        """
        result = prepare_code_source(src)
        assert "__result__ = result" in result
        assert result.endswith("__result__")

    def test_empty_source(self):
        result = prepare_code_source("")
        assert "__result__" in result

    def test_whitespace_only_source(self):
        result = prepare_code_source("   \n  \n  ")
        assert "__result__" in result

    def test_single_literal(self):
        result = prepare_code_source("42")
        assert "__result__" in result
        assert "42" in result

    def test_function_call_expression(self):
        result = prepare_code_source("add(x, y) * 2")
        assert "__result__" in result
        assert "add(x, y) * 2" in result


# ---------------------------------------------------------------------------
# _classify_refs
# ---------------------------------------------------------------------------


class TestClassifyRefs:
    def test_empty_dict(self):
        refs = _classify_refs({})
        assert refs == {"task_refs": {}, "trace_refs": {}, "durable_refs": {}}

    def test_task_template_classified(self):
        """TaskTemplate instances go into task_refs."""
        from flyte.sandbox import task

        @task
        def add(x: int, y: int) -> int:
            return x + y

        refs = _classify_refs({"add": add})
        assert "add" in refs["task_refs"]
        assert refs["trace_refs"] == {}
        assert refs["durable_refs"] == {}

    def test_plain_callable_classified_as_trace(self):
        """Plain callables default to trace_refs."""

        def my_func(x):
            return x

        refs = _classify_refs({"my_func": my_func})
        assert "my_func" in refs["trace_refs"]
        assert refs["task_refs"] == {}

    def test_mixed_refs(self):
        from flyte.sandbox import task

        @task
        def add(x: int, y: int) -> int:
            return x + y

        def helper(x):
            return x

        refs = _classify_refs({"add": add, "helper": helper})
        assert "add" in refs["task_refs"]
        assert "helper" in refs["trace_refs"]


# ---------------------------------------------------------------------------
# orchestrate() factory
# ---------------------------------------------------------------------------


class TestOrchestrateFactory:
    def test_creates_code_task_template(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        assert isinstance(t, CodeTaskTemplate)
        assert isinstance(t, SandboxedTaskTemplate)

    def test_default_name(self):
        t = orchestrate("x", inputs={"x": int}, output=int)
        assert t.name == "sandboxed-code"

    def test_custom_name(self):
        t = orchestrate("x", inputs={"x": int}, output=int, name="my-code")
        assert t.name == "my-code"

    def test_interface_from_types(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        assert "x" in t.interface.inputs
        assert "y" in t.interface.inputs
        assert "o0" in t.interface.outputs

    def test_interface_no_output(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int})
        assert t.interface.outputs == {}

    def test_source_code_populated(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        assert t._source_code != ""
        assert "__result__" in t._source_code

    def test_input_names_populated(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        assert t._input_names == ["x", "y"]

    def test_no_external_refs_pure(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        assert not t._has_external_refs

    def test_external_refs_with_tasks(self):
        from flyte.sandbox import task

        @task
        def add(x: int, y: int) -> int:
            return x + y

        t = orchestrate(
            "add(x, y)",
            inputs={"x": int, "y": int},
            output=int,
            tasks=[add],
        )
        assert t._has_external_refs
        assert "add" in t._external_refs["task_refs"]

    def test_task_type(self):
        t = orchestrate("x", inputs={"x": int}, output=int)
        assert t.task_type == "sandboxed-python"

    def test_default_plugin_config(self):
        t = orchestrate("x", inputs={"x": int}, output=int)
        assert isinstance(t.plugin_config, SandboxedConfig)
        assert t.plugin_config.timeout_ms == 30_000

    def test_custom_timeout(self):
        t = orchestrate("x", inputs={"x": int}, output=int, timeout_ms=5000)
        assert t.plugin_config.timeout_ms == 5000

    def test_retries(self):
        t = orchestrate("x", inputs={"x": int}, output=int, retries=3)
        assert t.retries.count == 3

    def test_forward_not_supported(self):
        t = orchestrate("x", inputs={"x": int}, output=int)
        with pytest.raises(NotImplementedError, match="does not support forward"):
            t.forward(x=1)

    def test_build_inputs(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        inputs = t._build_inputs(1, 2)
        assert inputs == {"x": 1, "y": 2}

    def test_build_inputs_kwargs(self):
        t = orchestrate("x + y", inputs={"x": int, "y": int}, output=int)
        inputs = t._build_inputs(x=1, y=2)
        assert inputs == {"x": 1, "y": 2}

    def test_rejects_unsupported_input_type(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="unsupported type"):
            orchestrate("x", inputs={"x": Custom}, output=int)

    def test_rejects_unsupported_output_type(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="unsupported type"):
            orchestrate("x", inputs={"x": int}, output=Custom)


# ---------------------------------------------------------------------------
# orchestrate_local() — requires pydantic-monty
# ---------------------------------------------------------------------------

try:
    from pydantic_monty import Monty  # noqa: F401

    HAS_MONTY = True
except ImportError:
    HAS_MONTY = False


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestRunPurePython:
    @pytest.mark.asyncio
    async def test_simple_expression(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local("x + y", inputs={"x": 1, "y": 2})
        assert result == 3

    @pytest.mark.asyncio
    async def test_assignment_return(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local("result = x * y", inputs={"x": 3, "y": 4})
        assert result == 12

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local(
            """
            a = x + 1
            b = y + 2
            a * b
            """,
            inputs={"x": 2, "y": 3},
        )
        assert result == 15  # (2+1) * (3+2)

    @pytest.mark.asyncio
    async def test_multiline_assignment(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local(
            """
            partial = x + y
            result = partial * 2
            """,
            inputs={"x": 1, "y": 2},
        )
        assert result == 6  # (1+2) * 2

    @pytest.mark.asyncio
    async def test_string_result(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local("name + ' world'", inputs={"name": "hello"})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_list_result(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local("[x, y, x + y]", inputs={"x": 1, "y": 2})
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# TaskEnvironment.sandbox.orchestrate()
# ---------------------------------------------------------------------------


class TestEnvironmentSandboxOrchestrate:
    def test_bare_decorator(self):
        """@env.sandbox.orchestrate without arguments."""
        import flyte

        env = flyte.TaskEnvironment(name="test-env")

        @env.sandbox.orchestrate
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, SandboxedTaskTemplate)
        assert add.name == "test-env.add"
        assert "test-env.add" in env.tasks

    def test_decorator_with_args(self):
        """@env.sandbox.orchestrate(timeout_ms=...) with arguments."""
        import flyte

        env = flyte.TaskEnvironment(name="test-env2")

        @env.sandbox.orchestrate(timeout_ms=5_000, retries=2)
        def multiply(x: int, y: int) -> int:
            return x * y

        assert isinstance(multiply, SandboxedTaskTemplate)
        assert multiply.plugin_config.timeout_ms == 5_000
        assert multiply.retries.count == 2
        assert "test-env2.multiply" in env.tasks

    def test_uses_environment_image(self):
        """Sandboxed task should use the environment's image, not a hardcoded one."""
        import flyte

        env = flyte.TaskEnvironment(name="img-test", image="my-custom-image:latest")

        @env.sandbox.orchestrate
        def noop(x: int) -> int:
            return x

        assert noop.image.base_image == "my-custom-image:latest"

    def test_parent_env_set(self):
        """Sandboxed task should have parent_env pointing to the environment."""
        import flyte

        env = flyte.TaskEnvironment(name="parent-test")

        @env.sandbox.orchestrate
        def sub(x: int) -> int:
            return x

        assert sub.parent_env() is env
        assert sub.parent_env_name == "parent-test"

    def test_rejects_async_function(self):
        """Sandboxed tasks must be synchronous."""
        import flyte

        env = flyte.TaskEnvironment(name="async-test")

        with pytest.raises(TypeError, match="must be synchronous"):

            @env.sandbox.orchestrate
            async def bad(x: int) -> int:
                return x

    def test_forward(self):
        """forward() should call the function directly."""
        import flyte

        env = flyte.TaskEnvironment(name="fwd-test")

        @env.sandbox.orchestrate
        def double(x: int) -> int:
            return x * 2

        assert double.forward(5) == 10

    def test_custom_name(self):
        """name= overrides the auto-generated name."""
        import flyte

        env = flyte.TaskEnvironment(name="name-test")

        @env.sandbox.orchestrate(name="my-custom-name")
        def thing(x: int) -> int:
            return x

        assert thing.name == "my-custom-name"
        assert "my-custom-name" in env.tasks
