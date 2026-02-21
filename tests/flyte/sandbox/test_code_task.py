"""Tests for flyte.sandbox.orchestrator(), flyte.sandbox.orchestrate_local(),
and TaskEnvironment.sandbox.orchestrator().
"""

import pytest

from flyte.sandbox import orchestrator
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
    def test_expression_passes_through(self):
        result = prepare_code_source("x + y")
        assert result == "x + y"

    def test_assignment_passes_through(self):
        """Assignments pass through as-is — no magic variable appending."""
        result = prepare_code_source("result = x + y")
        assert result == "result = x + y"

    def test_multiline_passes_through(self):
        src = """\
        a = x + 1
        b = a * 2
        b
        """
        result = prepare_code_source(src)
        assert "a = x + 1" in result
        assert "b = a * 2" in result
        lines = result.splitlines()
        assert lines[-1] == "b"

    def test_empty_source(self):
        result = prepare_code_source("")
        assert result == "None"

    def test_whitespace_only_source(self):
        result = prepare_code_source("   \n  \n  ")
        assert result == "None"

    def test_single_literal(self):
        result = prepare_code_source("42")
        assert result == "42"

    def test_function_call_expression(self):
        result = prepare_code_source("add(x, y) * 2")
        assert result == "add(x, y) * 2"

    def test_dedents_indented_source(self):
        src = """\
        x = 1
        x + 2
        """
        result = prepare_code_source(src)
        assert result == "x = 1\nx + 2"


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
# orchestrator() factory
# ---------------------------------------------------------------------------


class TestOrchestrateFactory:
    def test_creates_code_task_template(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        assert isinstance(t, CodeTaskTemplate)
        assert isinstance(t, SandboxedTaskTemplate)

    def test_default_name(self):
        t = orchestrator("x", inputs={"x": int}, output=int)
        assert t.name == "sandboxed-code"

    def test_custom_name(self):
        t = orchestrator("x", inputs={"x": int}, output=int, name="my-code")
        assert t.name == "my-code"

    def test_interface_from_types(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        assert "x" in t.interface.inputs
        assert "y" in t.interface.inputs
        assert "o0" in t.interface.outputs

    def test_interface_no_output(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int})
        assert t.interface.outputs == {}

    def test_source_code_populated(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        assert t._source_code != ""
        assert "x + y" in t._source_code

    def test_input_names_populated(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        assert t._input_names == ["x", "y"]

    def test_no_external_refs_pure(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        assert not t._has_external_refs

    def test_external_refs_with_tasks(self):
        from flyte.sandbox import task

        @task
        def add(x: int, y: int) -> int:
            return x + y

        t = orchestrator(
            "add(x, y)",
            inputs={"x": int, "y": int},
            output=int,
            tasks=[add],
        )
        assert t._has_external_refs
        assert "add" in t._external_refs["task_refs"]

    def test_task_type(self):
        t = orchestrator("x", inputs={"x": int}, output=int)
        assert t.task_type == "sandboxed-python"

    def test_default_plugin_config(self):
        t = orchestrator("x", inputs={"x": int}, output=int)
        assert isinstance(t.plugin_config, SandboxedConfig)
        assert t.plugin_config.timeout_ms == 30_000

    def test_custom_timeout(self):
        t = orchestrator("x", inputs={"x": int}, output=int, timeout_ms=5000)
        assert t.plugin_config.timeout_ms == 5000

    def test_retries(self):
        t = orchestrator("x", inputs={"x": int}, output=int, retries=3)
        assert t.retries.count == 3

    def test_forward_not_supported(self):
        t = orchestrator("x", inputs={"x": int}, output=int)
        with pytest.raises(NotImplementedError, match="does not support forward"):
            t.forward(x=1)

    def test_build_inputs(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        inputs = t._build_inputs(1, 2)
        assert inputs == {"x": 1, "y": 2}

    def test_build_inputs_kwargs(self):
        t = orchestrator("x + y", inputs={"x": int, "y": int}, output=int)
        inputs = t._build_inputs(x=1, y=2)
        assert inputs == {"x": 1, "y": 2}

    def test_rejects_unsupported_input_type(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="unsupported type"):
            orchestrator("x", inputs={"x": Custom}, output=int)

    def test_rejects_unsupported_output_type(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="unsupported type"):
            orchestrator("x", inputs={"x": int}, output=Custom)


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
    async def test_expression_return(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local("x * y", inputs={"x": 3, "y": 4})
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
    async def test_multiline_last_expression(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local(
            """
            partial = x + y
            partial * 2
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
# TaskEnvironment.sandbox.orchestrator()
# ---------------------------------------------------------------------------


class TestEnvironmentSandboxOrchestrator:
    def test_bare_decorator(self):
        """@env.sandbox.orchestrator without arguments."""
        import flyte

        env = flyte.TaskEnvironment(name="test-env")

        @env.sandbox.orchestrator
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, SandboxedTaskTemplate)
        assert add.name == "test-env.add"
        assert "test-env.add" in env.tasks

    def test_decorator_with_args(self):
        """@env.sandbox.orchestrator(timeout_ms=...) with arguments."""
        import flyte

        env = flyte.TaskEnvironment(name="test-env2")

        @env.sandbox.orchestrator(timeout_ms=5_000, retries=2)
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

        @env.sandbox.orchestrator
        def noop(x: int) -> int:
            return x

        assert noop.image.base_image == "my-custom-image:latest"

    def test_parent_env_set(self):
        """Sandboxed task should have parent_env pointing to the environment."""
        import flyte

        env = flyte.TaskEnvironment(name="parent-test")

        @env.sandbox.orchestrator
        def sub(x: int) -> int:
            return x

        assert sub.parent_env() is env
        assert sub.parent_env_name == "parent-test"

    def test_accepts_async_function(self):
        """Sandboxed tasks support async def — Monty handles async natively."""
        import flyte

        env = flyte.TaskEnvironment(name="async-test")

        @env.sandbox.orchestrator
        async def async_orch(x: int) -> int:
            return x

        assert isinstance(async_orch, SandboxedTaskTemplate)
        assert "async def async_orch" in async_orch._source_code

    def test_forward(self):
        """forward() should call the function directly."""
        import flyte

        env = flyte.TaskEnvironment(name="fwd-test")

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        assert double.forward(5) == 10

    def test_custom_name(self):
        """name= overrides the auto-generated name."""
        import flyte

        env = flyte.TaskEnvironment(name="name-test")

        @env.sandbox.orchestrator(name="my-custom-name")
        def thing(x: int) -> int:
            return x

        assert thing.name == "my-custom-name"
        assert "my-custom-name" in env.tasks

    def test_code_string_mode(self):
        """env.sandbox.orchestrator with a code string creates CodeTaskTemplate."""
        import flyte
        from flyte.sandbox import task

        @task
        def add(x: int, y: int) -> int:
            return x + y

        env = flyte.TaskEnvironment(name="code-str-test")
        t = env.sandbox.orchestrator(
            "add(x, y)",
            inputs={"x": int, "y": int},
            output=int,
            tasks=[add],
        )
        assert isinstance(t, CodeTaskTemplate)
        assert t.name == "sandboxed-code"


# ---------------------------------------------------------------------------
# Monty return support — requires pydantic-monty
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestMontyReturnSupport:
    """Prove that Monty natively supports return statements in function defs."""

    def test_function_def_with_return(self):
        """Function def with return + trailing call returns correct value."""
        from pydantic_monty import Monty

        code = """\
def add(x, y):
    return x + y
add(x, y)
"""
        monty = Monty(code, inputs=["x", "y"])
        result = monty.run(inputs={"x": 3, "y": 4})
        assert result == 7

    def test_conditional_returns(self):
        """Conditional returns (if/elif/else) each branch works."""
        from pydantic_monty import Monty

        code = """\
def classify(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
classify(x)
"""
        monty = Monty(code, inputs=["x"])
        assert monty.run(inputs={"x": 5}) == "positive"
        assert monty.run(inputs={"x": -3}) == "negative"
        assert monty.run(inputs={"x": 0}) == "zero"

    def test_bare_return(self):
        """Bare return (no value) returns None."""
        from pydantic_monty import Monty

        code = """\
def noop(x):
    if x > 0:
        return
    return x
noop(x)
"""
        monty = Monty(code, inputs=["x"])
        assert monty.run(inputs={"x": 5}) is None
        assert monty.run(inputs={"x": -1}) == -1

    def test_function_no_return(self):
        """Function with no return returns None."""
        from pydantic_monty import Monty

        code = """\
def noop(x):
    _ = x + 1
noop(x)
"""
        monty = Monty(code, inputs=["x"])
        assert monty.run(inputs={"x": 42}) is None

    def test_function_with_external_functions(self):
        """Function def with external calls uses MontySnapshot for pause/resume."""
        from pydantic_monty import MontyComplete, MontySnapshot

        code = """\
def pipeline(x):
    doubled = double(x)
    return doubled + 1
pipeline(x)
"""
        from pydantic_monty import Monty

        monty = Monty(code, inputs=["x"], external_functions=["double"])
        snapshot = monty.start(inputs={"x": 5})
        assert isinstance(snapshot, MontySnapshot)
        assert snapshot.function_name == "double"
        assert snapshot.args == (5,)

        # Resume with the result of the external function
        result = snapshot.resume(return_value=10)  # double(5) = 10
        assert isinstance(result, MontyComplete)
        assert result.output == 11  # 10 + 1


# ---------------------------------------------------------------------------
# orchestrate_local with function defs — requires pydantic-monty
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestRunPurePythonFunctionDefs:
    @pytest.mark.asyncio
    async def test_function_def_with_return(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local(
            """
            def compute(x, y):
                return x * y + 1
            compute(x, y)
            """,
            inputs={"x": 3, "y": 4},
        )
        assert result == 13  # 3 * 4 + 1

    @pytest.mark.asyncio
    async def test_conditional_returns(self):
        from flyte.sandbox import orchestrate_local

        result = await orchestrate_local(
            """
            def classify(x):
                if x > 0:
                    return "positive"
                else:
                    return "non-positive"
            classify(x)
            """,
            inputs={"x": 5},
        )
        assert result == "positive"

    @pytest.mark.asyncio
    async def test_with_external_task(self):
        from flyte.sandbox import orchestrate_local, task

        @task
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            "double(x) + 1",
            inputs={"x": 5},
            tasks=[double],
        )
        assert result == 11  # 5 * 2 + 1


# ---------------------------------------------------------------------------
# Sandbox IO types — validates interface accepts File/Dir/DataFrame
# ---------------------------------------------------------------------------


class TestSandboxIOTypes:
    def test_task_with_file_input_output(self):
        """@sandbox.task with File input/output type annotations validates."""
        from flyte.io import File
        from flyte.sandbox import task

        @task
        def process(f: File) -> File:
            return f

        assert isinstance(process, SandboxedTaskTemplate)
        assert "f" in process.interface.inputs
        assert "o0" in process.interface.outputs

    def test_task_with_dir_input_output(self):
        """@sandbox.task with Dir input/output type annotations validates."""
        from flyte.io import Dir
        from flyte.sandbox import task

        @task
        def process(d: Dir) -> Dir:
            return d

        assert isinstance(process, SandboxedTaskTemplate)
        assert "d" in process.interface.inputs
        assert "o0" in process.interface.outputs

    def test_task_with_dataframe_input_output(self):
        """@sandbox.task with DataFrame input/output type annotations validates."""
        from flyte.io import DataFrame
        from flyte.sandbox import task

        @task
        def process(df: DataFrame) -> DataFrame:
            return df

        assert isinstance(process, SandboxedTaskTemplate)
        assert "df" in process.interface.inputs
        assert "o0" in process.interface.outputs

    def test_orchestrator_with_file_types(self):
        """orchestrator() code string with File in inputs/output validates."""
        from flyte.io import File

        t = orchestrator("f", inputs={"f": File}, output=File)
        assert isinstance(t, CodeTaskTemplate)
        assert "f" in t.interface.inputs
        assert "o0" in t.interface.outputs

    def test_orchestrator_with_dir_types(self):
        """orchestrator() code string with Dir in inputs/output validates."""
        from flyte.io import Dir

        t = orchestrator("d", inputs={"d": Dir}, output=Dir)
        assert isinstance(t, CodeTaskTemplate)
        assert "d" in t.interface.inputs
        assert "o0" in t.interface.outputs

    def test_orchestrator_with_dataframe_types(self):
        """orchestrator() code string with DataFrame in inputs/output validates."""
        from flyte.io import DataFrame

        t = orchestrator("df", inputs={"df": DataFrame}, output=DataFrame)
        assert isinstance(t, CodeTaskTemplate)
        assert "df" in t.interface.inputs
        assert "o0" in t.interface.outputs

    def test_env_orchestrator_with_file_types(self):
        """env.sandbox.orchestrator decorated function with File annotations validates."""
        import flyte
        from flyte.io import File

        env = flyte.TaskEnvironment(name="io-file-test")

        @env.sandbox.orchestrator
        def passthrough(f: File) -> File:
            return f

        assert isinstance(passthrough, SandboxedTaskTemplate)
        assert "f" in passthrough.interface.inputs
        assert "o0" in passthrough.interface.outputs

    def test_env_orchestrator_with_dir_types(self):
        """env.sandbox.orchestrator decorated function with Dir annotations validates."""
        import flyte
        from flyte.io import Dir

        env = flyte.TaskEnvironment(name="io-dir-test")

        @env.sandbox.orchestrator
        def passthrough(d: Dir) -> Dir:
            return d

        assert isinstance(passthrough, SandboxedTaskTemplate)
        assert "d" in passthrough.interface.inputs
        assert "o0" in passthrough.interface.outputs

    def test_env_orchestrator_with_dataframe_types(self):
        """env.sandbox.orchestrator decorated function with DataFrame annotations validates."""
        import flyte
        from flyte.io import DataFrame

        env = flyte.TaskEnvironment(name="io-df-test")

        @env.sandbox.orchestrator
        def passthrough(df: DataFrame) -> DataFrame:
            return df

        assert isinstance(passthrough, SandboxedTaskTemplate)
        assert "df" in passthrough.interface.inputs
        assert "o0" in passthrough.interface.outputs
