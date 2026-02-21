from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict

from flyte.io import DataFrame, Dir, File

# Tag used to identify marshaled IO types inside Monty
_IO_TYPE_KEY = "__flyte_io_type__"

_IO_TYPES = {"File": File, "Dir": Dir, "DataFrame": DataFrame}


def _to_monty(value: Any) -> Any:
    """Marshal a flyte.io type to a dict Monty can hold."""
    if isinstance(value, File):
        return {_IO_TYPE_KEY: "File", **value.model_dump()}
    if isinstance(value, Dir):
        return {_IO_TYPE_KEY: "Dir", **value.model_dump()}
    if isinstance(value, DataFrame):
        return {_IO_TYPE_KEY: "DataFrame", **value.model_dump()}
    if isinstance(value, list):
        return [_to_monty(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_monty(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_monty(v) for v in value)
    return value


def _from_monty(value: Any) -> Any:
    """Unmarshal a tagged dict back to a flyte.io type."""
    if isinstance(value, dict) and _IO_TYPE_KEY in value:
        tag = value[_IO_TYPE_KEY]
        cls = _IO_TYPES.get(tag)
        if cls is not None:
            payload = {k: v for k, v in value.items() if k != _IO_TYPE_KEY}
            return cls.model_validate(payload)  # type: ignore[attr-defined]
    if isinstance(value, dict):
        return {k: _from_monty(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_monty(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_from_monty(v) for v in value)
    return value


class ExternalFunctionBridge:
    """Drives Monty execution with external function dispatch.

    Uses the low-level ``Monty.start()`` / ``MontySnapshot.resume()`` loop,
    awaiting each external call before resuming. This ensures async external
    functions (task.aio, durable ops) are properly resolved.
    """

    def __init__(
        self,
        task_refs: Dict[str, Any],
        trace_refs: Dict[str, Any],
        durable_refs: Dict[str, Any],
    ) -> None:
        self.task_refs = task_refs
        self.trace_refs = trace_refs
        self.durable_refs = durable_refs
        self._all_refs: Dict[str, Any] = {}
        self._all_refs.update(task_refs)
        self._all_refs.update(trace_refs)
        self._all_refs.update(durable_refs)

    def _build_external_functions(self) -> Dict[str, Callable]:
        """Build async callables for each external ref."""
        from flyte._task import TaskTemplate

        result: Dict[str, Callable] = {}
        for name, ref in self._all_refs.items():
            if isinstance(ref, TaskTemplate):
                result[name] = ref.aio
            elif callable(ref):
                result[name] = ref
            else:
                raise RuntimeError(f"External ref '{name}' is not callable")
        return result

    async def execute_monty(self, monty_cls: Any, code: str, input_names: list[str], inputs: Dict[str, Any]) -> Any:
        """Run *code* in Monty, awaiting each external call before resuming.

        Parameters
        ----------
        monty_cls:
            The ``pydantic_monty.Monty`` class.
        code:
            The rewritten function body source.
        input_names:
            List of input parameter names (declared at compile time).
        inputs:
            Mapping of input parameter names to values (provided at run time).
        """
        import inspect

        from pydantic_monty import MontyComplete, MontySnapshot

        ext_names = list(self._all_refs.keys())
        monty = monty_cls(code, inputs=input_names, external_functions=ext_names)
        ext_fns = self._build_external_functions()

        # Marshal any IO types in the initial inputs so Monty can hold them
        monty_inputs = {k: _to_monty(v) for k, v in inputs.items()}

        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor() as pool:

            async def run_in_pool(func):
                return await loop.run_in_executor(pool, func)

            progress = await run_in_pool(partial(monty.start, inputs=monty_inputs))

            while True:
                if isinstance(progress, MontyComplete):
                    return _from_monty(progress.output)
                elif isinstance(progress, MontySnapshot):
                    fn = ext_fns.get(progress.function_name)
                    if fn is None:
                        raise RuntimeError(f"Sandboxed task called unknown external function: {progress.function_name}")

                    # Unmarshal IO handles before calling the external function
                    args = [_from_monty(a) for a in progress.args]
                    kwargs = {k: _from_monty(v) for k, v in progress.kwargs.items()}

                    # Call the external function and await if async.
                    # Loop because TaskTemplate.aio() in local mode may return
                    # an unawaited coroutine from forward() for async functions.
                    result = fn(*args, **kwargs)
                    while inspect.iscoroutine(result):
                        result = await result

                    # Marshal IO types so Monty can hold the return value
                    progress = await run_in_pool(partial(progress.resume, return_value=_to_monty(result)))
                else:
                    raise RuntimeError(f"Unexpected Monty progress state: {progress!r}")
