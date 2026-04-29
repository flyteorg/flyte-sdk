from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List

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


def _resume_monty_snapshot(snapshot: Any, return_value: Any) -> Any:
    """Pass a tool return value into ``FunctionSnapshot.resume``.

    pydantic-monty 0.0.17+ expects ``resume({"return_value": ...})`` (an
    ``ExternalResult`` dict). Older releases used ``resume(return_value=...)``
    only.

    Without this wrapper, returning a plain dict from a tool (e.g.
    ``{"summary": "..."}``) is mistaken for an ``ExternalResult`` and Monty
    raises about invalid ``ExternalResult`` keys.
    """
    marshaled = _to_monty(return_value)
    sig = inspect.signature(snapshot.resume)
    if "result" in sig.parameters:
        return snapshot.resume({"return_value": marshaled})
    return snapshot.resume(return_value=marshaled)


class ExternalFunctionBridge:
    """Drives Monty execution with external function dispatch.

    Uses the low-level `Monty.start()` / `FunctionSnapshot.resume()` loop,
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

    async def _handle_flyte_map(
        self,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> List[Any]:
        """Handle a ``flyte_map("task_name", *iterables, **kwargs)`` call.

        Resolves the task name to the real ``TaskTemplate``, then delegates
        to ``flyte.map.aio`` so that concurrency, group tracking, and
        ``return_exceptions`` all work identically to top-level ``flyte.map``.
        """
        from flyte._map import map as flyte_map

        if len(args) < 2:
            raise RuntimeError("flyte_map requires at least 2 arguments: flyte_map(task_name, iterable, ...)")

        task_name = args[0]
        if not isinstance(task_name, str):
            raise RuntimeError(f"flyte_map first argument must be a task name string, got {type(task_name).__name__}")

        task = self._all_refs.get(task_name)
        if task is None:
            available = ", ".join(sorted(self._all_refs.keys())) or "(none)"
            raise RuntimeError(f"flyte_map: task '{task_name}' not found in registered tasks. Available: {available}")

        iterables = [_from_monty(a) for a in args[1:]]

        # Forward kwargs that flyte.map.aio accepts
        map_kwargs: Dict[str, Any] = {}
        for key in ("group_name", "concurrency", "return_exceptions"):
            if key in kwargs:
                map_kwargs[key] = kwargs[key]

        results: List[Any] = []
        async for r in flyte_map.aio(task, *iterables, **map_kwargs):
            results.append(r)
        return results

    async def execute_monty(self, monty_cls: Any, code: str, input_names: list[str], inputs: Dict[str, Any]) -> Any:
        """Run *code* in Monty, awaiting each external call before resuming.

        Parameters
        ----------
        monty_cls:
            The `pydantic_monty.Monty` class.
        code:
            The rewritten function body source.
        input_names:
            List of input parameter names (declared at compile time).
        inputs:
            Mapping of input parameter names to values (provided at run time).
        """
        from pydantic_monty import FunctionSnapshot, MontyComplete

        monty = monty_cls(code, inputs=input_names)
        ext_fns = self._build_external_functions()

        # Marshal any IO types in the initial inputs so Monty can hold them
        monty_inputs = {k: _to_monty(v) for k, v in inputs.items()}

        progress = monty.start(inputs=monty_inputs)

        while True:
            if isinstance(progress, MontyComplete):
                return _from_monty(progress.output)
            elif isinstance(progress, FunctionSnapshot):
                # Handle flyte_map as a special built-in for parallel execution
                if progress.function_name == "flyte_map":
                    args = [_from_monty(a) for a in progress.args]
                    kwargs = {k: _from_monty(v) for k, v in progress.kwargs.items()}
                    result = await self._handle_flyte_map(args, kwargs)
                    progress = _resume_monty_snapshot(progress, result)
                    continue

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

                progress = _resume_monty_snapshot(progress, result)
            else:
                raise RuntimeError(f"Unexpected Monty progress state: {progress!r}")
