"""Tests for the sandboxed task decorator and SandboxedTaskTemplate."""

from typing import Dict, List, Optional

import pytest

from flyte.sandbox import task
from flyte.sandbox._config import SandboxedConfig
from flyte.sandbox._task import SandboxedTaskTemplate


class TestTaskDecorator:
    def test_bare_decorator(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, SandboxedTaskTemplate)
        assert add.name == "TestTaskDecorator.test_bare_decorator.<locals>.add"
        assert add.task_type == "sandboxed-python"

    def test_decorator_with_args(self):
        @task(timeout_ms=5000, name="my_multiply")
        def multiply(x: int, y: int) -> int:
            return x * y

        assert isinstance(multiply, SandboxedTaskTemplate)
        assert multiply.name == "my_multiply"
        assert multiply.plugin_config.timeout_ms == 5000

    def test_decorator_with_cache(self):
        @task(cache="auto")
        def cached_fn(x: int) -> int:
            return x * 2

        assert isinstance(cached_fn, SandboxedTaskTemplate)

    def test_decorator_with_retries(self):
        @task(retries=3)
        def retried_fn(x: int) -> int:
            return x

        assert isinstance(retried_fn, SandboxedTaskTemplate)
        assert retried_fn.retries.count == 3


class TestSandboxedTaskTemplate:
    def test_rejects_async_function(self):
        with pytest.raises(TypeError, match="must be synchronous"):

            @task
            async def bad_fn(x: int) -> int:
                return x

    def test_rejects_unsupported_types(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="unsupported type"):

            @task
            def bad_fn(x: Custom) -> int:
                return 0

    def test_source_extraction(self):
        @task
        def fn(x: int) -> int:
            return x + 1

        assert fn._source_code != ""
        assert fn._input_names == ["x"]

    def test_default_config(self):
        @task
        def fn(x: int) -> int:
            return x

        assert fn.plugin_config == SandboxedConfig()
        assert fn.plugin_config.timeout_ms == 30_000
        assert fn.plugin_config.max_memory == 50 * 1024 * 1024

    def test_custom_config(self):
        @task(timeout_ms=1000, max_memory=1024, max_stack_depth=128, type_check=False)
        def fn(x: int) -> int:
            return x

        assert fn.plugin_config.timeout_ms == 1000
        assert fn.plugin_config.max_memory == 1024
        assert fn.plugin_config.max_stack_depth == 128
        assert fn.plugin_config.type_check is False

    def test_forward_bypasses_sandbox(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        # forward() calls the original function directly
        assert add.forward(2, 3) == 5

    def test_interface_types(self):
        @task
        def fn(a: int, b: str, c: List[int]) -> Dict[str, int]:
            return {}

        assert "a" in fn.interface.inputs
        assert "b" in fn.interface.inputs
        assert "c" in fn.interface.inputs

    def test_no_external_refs_for_pure_python(self):
        @task
        def pure(x: int) -> int:
            return x * 2

        assert not pure._has_external_refs

    def test_image_defaults_to_auto(self):
        @task
        def fn(x: int) -> int:
            return x

        # image="auto" gets resolved to Image.from_debian_base() in __post_init__
        from flyte._image import Image

        assert isinstance(fn.image, Image)

    def test_optional_type(self):
        @task
        def fn(x: Optional[int]) -> Optional[str]:
            if x is None:
                return None
            return str(x)

        assert isinstance(fn, SandboxedTaskTemplate)

    def test_build_inputs(self):
        @task
        def fn(a: int, b: str) -> int:
            return 0

        inputs = fn._build_inputs(1, b="hello")
        assert inputs == {"a": 1, "b": "hello"}

    def test_build_inputs_positional(self):
        @task
        def fn(a: int, b: str) -> int:
            return 0

        inputs = fn._build_inputs(1, "hello")
        assert inputs == {"a": 1, "b": "hello"}
