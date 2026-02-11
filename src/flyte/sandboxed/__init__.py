"""Sandboxed tasks powered by Monty (Pydantic's Rust-based sandboxed Python interpreter).

Sandboxed tasks are:
- **Side-effect free**: No filesystem, network, or OS access
- **Super fast**: Microsecond startup for pure Python
- **Multiplexable**: Many tasks run safely on the same Python process

Usage::

    from flyte import sandboxed

    @sandboxed.task
    def add(x: int, y: int) -> int:
        return x + y

    @sandboxed.task(timeout_ms=5000)
    def multiply(x: int, y: int) -> int:
        return x * y

    # Create a reusable task from a code string
    pipeline = sandboxed.code(
        "add(x, y) * 2",
        inputs={"x": int, "y": int},
        output=int,
        functions={"add": add},
    )

    # One-shot execution of a code string
    result = await sandboxed.run(
        "x + y",
        inputs={"x": 1, "y": 2},
    )
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Union, overload

from flyte._cache import CacheRequest
from flyte.models import NativeInterface

from ._code_task import CodeTaskTemplate, _classify_refs, _prepare_code_source
from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate, _lazy_import_monty

__all__ = [
    "CodeTaskTemplate",
    "SandboxedConfig",
    "SandboxedTaskTemplate",
    "code",
    "run",
    "task",
]


@overload
def task(_func: Callable, /) -> SandboxedTaskTemplate: ...


@overload
def task(
    *,
    timeout_ms: int = 30_000,
    max_memory: int = 50 * 1024 * 1024,
    max_stack_depth: int = 256,
    type_check: bool = True,
    name: Optional[str] = None,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Callable[[Callable], SandboxedTaskTemplate]: ...


def task(
    _func: Optional[Callable] = None,
    /,
    *,
    timeout_ms: int = 30_000,
    max_memory: int = 50 * 1024 * 1024,
    max_stack_depth: int = 256,
    type_check: bool = True,
    name: Optional[str] = None,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Union[SandboxedTaskTemplate, Callable[[Callable], SandboxedTaskTemplate]]:
    """Decorator to create a sandboxed Flyte task.

    Can be used with or without arguments::

        @sandboxed.task
        def add(x: int, y: int) -> int:
            return x + y

        @sandboxed.task(timeout_ms=5000)
        def multiply(x: int, y: int) -> int:
            return x * y
    """
    config = SandboxedConfig(
        max_memory=max_memory,
        max_stack_depth=max_stack_depth,
        timeout_ms=timeout_ms,
        type_check=type_check,
    )

    def decorator(func: Callable) -> SandboxedTaskTemplate:
        from flyte._image import Image

        image = Image.from_debian_base().with_pip_packages("pydantic-monty")

        interface = NativeInterface.from_callable(func)
        return SandboxedTaskTemplate(
            func=func,
            name=name or func.__qualname__,
            interface=interface,
            plugin_config=config,
            image=image,
            cache=cache,
            retries=retries,
        )

    if _func is not None:
        return decorator(_func)
    return decorator


def code(
    source: str,
    *,
    inputs: Dict[str, type],
    output: type = type(None),
    functions: Optional[Dict[str, Any]] = None,
    name: str = "sandboxed-code",
    timeout_ms: int = 30_000,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> CodeTaskTemplate:
    """Create a reusable sandboxed task from a code string.

    The returned ``CodeTaskTemplate`` can be passed to ``flyte.run()``
    just like a decorated task.

    The **last expression** in *source* becomes the return value::

        pipeline = sandboxed.code(
            "add(x, y) * 2",
            inputs={"x": int, "y": int},
            output=int,
            functions={"add": add},
        )
        result = flyte.run(pipeline, x=1, y=2)  # → 6

    Parameters
    ----------
    source:
        Python code string to execute in the sandbox.
    inputs:
        Mapping of input names to their types.
    output:
        The return type (default ``NoneType``).
    functions:
        External functions (tasks, durable ops) available inside the sandbox.
    name:
        Task name (default ``"sandboxed-code"``).
    timeout_ms:
        Sandbox execution timeout in milliseconds.
    cache:
        Cache policy for the task.
    retries:
        Number of retries on failure.
    """
    from flyte._image import Image

    functions = functions or {}

    source_code = _prepare_code_source(source)
    input_names = list(inputs.keys())

    # Build NativeInterface from the explicit type dicts
    input_types = {k: (tp, inspect.Parameter.empty) for k, tp in inputs.items()}
    output_types = {"o0": output} if output is not type(None) else {}
    interface = NativeInterface(inputs=input_types, outputs=output_types)

    config = SandboxedConfig(timeout_ms=timeout_ms)
    image = Image.from_debian_base().with_pip_packages("pydantic-monty")

    # Dummy func for AsyncFunctionTaskTemplate compatibility
    dummy_func = lambda **kwargs: None  # noqa: E731

    return CodeTaskTemplate(
        func=dummy_func,
        name=name,
        interface=interface,
        plugin_config=config,
        image=image,
        cache=cache,
        retries=retries,
        _user_source=source_code,
        _user_input_names=input_names,
        _user_functions=functions,
    )


async def run(
    source: str,
    *,
    inputs: Dict[str, Any],
    functions: Optional[Dict[str, Any]] = None,
    timeout_ms: int = 30_000,
) -> Any:
    """One-shot execution of a code string in the Monty sandbox.

    Sends the code + inputs to Monty and returns the result directly,
    without creating a ``TaskTemplate`` or going through the controller.

    The **last expression** in *source* becomes the return value::

        result = await sandboxed.run(
            "add(x, y) * 2",
            inputs={"x": 1, "y": 2},
            functions={"add": add},
        )
        # → 6

    Parameters
    ----------
    source:
        Python code string to execute in the sandbox.
    inputs:
        Mapping of input names to their values.
    functions:
        External functions (tasks, durable ops) available inside the sandbox.
    timeout_ms:
        Sandbox execution timeout in milliseconds.
    """
    Monty = _lazy_import_monty()

    source_code = _prepare_code_source(source)
    input_names = list(inputs.keys())
    functions = functions or {}

    if not functions:
        # Pure Python — fast path, no external calls
        monty = Monty(source_code, inputs=input_names)
        return monty.run(inputs=inputs)
    else:
        from ._bridge import ExternalFunctionBridge

        refs = _classify_refs(functions)
        bridge = ExternalFunctionBridge(**refs)
        return await bridge.execute_monty(Monty, source_code, input_names, inputs)
