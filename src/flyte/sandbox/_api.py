"""Core public API for the sandbox module.

Provides ``orchestrator``, ``orchestrate_local``, and ``task``.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Union, overload

from flyte._cache import CacheRequest
from flyte._task import TaskTemplate
from flyte.models import NativeInterface

from ._code_task import CodeTaskTemplate, _classify_refs
from ._config import SandboxedConfig
from ._source import prepare_code_source
from ._task import SandboxedTaskTemplate, _lazy_import_monty


def _tasks_to_dict(tasks: List[Any]) -> Dict[str, Any]:
    """Convert a list of tasks/callables to a name→object dict.

    Raises ``ValueError`` on duplicate names.
    """
    result: Dict[str, Any] = {}
    for t in tasks:
        if isinstance(t, TaskTemplate):
            name = getattr(t, "func").__name__
        elif callable(t):
            name = t.__name__
        else:
            raise TypeError(f"Expected a callable or TaskTemplate, got {type(t)}")
        if name in result:
            raise ValueError(f"Duplicate task name '{name}'")
        result[name] = t
    return result


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

    .. warning:: Experimental feature: alpha — APIs may change without notice.

    Can be used with or without arguments::

        @sandbox.task
        def add(x: int, y: int) -> int:
            return x + y

        @sandbox.task(timeout_ms=5000)
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


def _orchestrator_impl(
    source: str,
    *,
    inputs: Dict[str, type],
    output: type = type(None),
    tasks: Optional[List[Any]] = None,
    name: str = "sandboxed-code",
    timeout_ms: int = 30_000,
    cache: CacheRequest = "disable",
    retries: int = 0,
    image: Optional[Any] = None,
    caller_module: str = "__main__",
) -> CodeTaskTemplate:
    """Internal implementation — use ``orchestrator()`` or ``env.sandbox.orchestrator()``."""
    from flyte._image import Image

    functions = _tasks_to_dict(tasks) if tasks else {}

    source_code = prepare_code_source(source)
    input_names = list(inputs.keys())

    # Build NativeInterface from the explicit type dicts
    input_types = {k: (tp, inspect.Parameter.empty) for k, tp in inputs.items()}
    output_types = {"o0": output} if output is not type(None) else {}
    interface = NativeInterface(inputs=input_types, outputs=output_types)

    config = SandboxedConfig(timeout_ms=timeout_ms)
    if image is None:
        image = Image.from_debian_base().with_pip_packages("pydantic-monty")

    # Dummy func for AsyncFunctionTaskTemplate compatibility
    dummy_func = lambda **kwargs: None  # noqa: E731
    dummy_func.__module__ = caller_module

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


def orchestrator(
    source: str,
    *,
    inputs: Dict[str, type],
    output: type = type(None),
    tasks: Optional[List[Any]] = None,
    name: str = "sandboxed-code",
    timeout_ms: int = 30_000,
    cache: CacheRequest = "disable",
    retries: int = 0,
    image: Optional[Any] = None,
) -> CodeTaskTemplate:
    """Create a reusable sandboxed task from a code string.

    .. warning:: Experimental feature: alpha — APIs may change without notice.

    The returned ``CodeTaskTemplate`` can be passed to ``flyte.run()``
    just like a decorated task.

    The **last expression** in *source* becomes the return value::

        pipeline = sandbox.orchestrator(
            "add(x, y) * 2",
            inputs={"x": int, "y": int},
            output=int,
            tasks=[add],
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
    tasks:
        List of external functions (tasks, durable ops) available inside the
        sandbox. Each item's ``__name__`` is used as the key.
    name:
        Task name (default ``"sandboxed-code"``).
    timeout_ms:
        Sandbox execution timeout in milliseconds.
    cache:
        Cache policy for the task.
    retries:
        Number of retries on failure.
    image:
        Docker image to use. If not provided, a default Debian image with
        ``pydantic-monty`` is created automatically.
    """
    return _orchestrator_impl(
        source,
        inputs=inputs,
        output=output,
        tasks=tasks,
        name=name,
        timeout_ms=timeout_ms,
        cache=cache,
        retries=retries,
        image=image,
        caller_module=sys._getframe(1).f_globals.get("__name__", "__main__"),
    )


async def orchestrate_local(
    source: str,
    *,
    inputs: Dict[str, Any],
    tasks: Optional[List[Any]] = None,
    timeout_ms: int = 30_000,
) -> Any:
    """One-shot local execution of a code string in the Monty sandbox.

    .. warning:: Experimental feature: alpha — APIs may change without notice.

    Sends the code + inputs to Monty and returns the result directly,
    without creating a ``TaskTemplate`` or going through the controller.

    The **last expression** in *source* becomes the return value::

        result = await sandbox.orchestrate_local(
            "add(x, y) * 2",
            inputs={"x": 1, "y": 2},
            tasks=[add],
        )
        # → 6

    Parameters
    ----------
    source:
        Python code string to execute in the sandbox.
    inputs:
        Mapping of input names to their values.
    tasks:
        List of external functions (tasks, durable ops) available inside the
        sandbox. Each item's ``__name__`` is used as the key.
    timeout_ms:
        Sandbox execution timeout in milliseconds.
    """
    Monty = _lazy_import_monty()

    source_code = prepare_code_source(source)
    input_names = list(inputs.keys())
    functions = _tasks_to_dict(tasks) if tasks else {}

    if not functions:
        # Pure Python — fast path, no external calls
        monty = Monty(source_code, inputs=input_names)
        return monty.run(inputs=inputs)
    else:
        from ._bridge import ExternalFunctionBridge

        refs = _classify_refs(functions)
        bridge = ExternalFunctionBridge(**refs)
        return await bridge.execute_monty(Monty, source_code, input_names, inputs)
