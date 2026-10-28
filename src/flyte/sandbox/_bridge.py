from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from flyte.io import DataFrame, Dir, File

if TYPE_CHECKING:
    from ._config import SandboxedConfig

# Tag used to identify marshaled IO types inside Monty
_IO_TYPE_KEY = "__flyte_io_type__"

_IO_TYPES = {"File": File, "Dir": Dir, "DataFrame": DataFrame}

# OS calls Monty hands to the host to wait out `time.sleep` / `asyncio.sleep`.
_SLEEP_FUNCTIONS = ("system.sleep", "system.async_sleep")


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


class RaiseInSandbox(Exception):
    """Raised by an external function to raise *exception* in the sandbox at the call,
    where the source can catch it, instead of failing the run."""

    def __init__(self, exception: BaseException):
        super().__init__(str(exception))
        self.exception = exception


async def _call_external(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call an external ref from the bridge's event loop, resolving to a concrete value.

    Coroutine functions are called inline and awaited. Everything else runs in a
    worker thread via `asyncio.to_thread` (which propagates contextvars, so the
    Flyte task context survives): a sync external may block — in particular a sync
    `@flyte.trace` wrapper makes blocking syncify calls, and when this loop *is*
    the syncify loop (e.g. agent code mode entered through a syncified call), an
    inline call would trip syncify's deadlock detection and kill the tool call.
    Offloading also keeps slow sync tools from stalling the loop.

    The trailing loop unwraps coroutines returned by non-coroutine-function
    callables — e.g. `TaskTemplate.aio()` in local mode may return an unawaited
    coroutine from `forward()`.
    """
    if inspect.iscoroutinefunction(fn):
        result = fn(*args, **kwargs)
    else:
        result = await asyncio.to_thread(fn, *args, **kwargs)
    while inspect.iscoroutine(result):
        result = await result
    return result


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


class _Worker:
    """The pooled Monty worker a feed runs on, which it gives back while it waits on tasks.

    An orchestrator that waits on its children would otherwise hold a worker
    for as long as they run, and children that are orchestrators need workers of
    their own: with more of them waiting than the pool has workers, none of the
    children could start.
    """

    def __init__(self, config: Optional["SandboxedConfig"]) -> None:
        self._config = config
        self.session: Any = None
        # Monty does not carry the sleep budget used in a dumped snapshot, so
        # the worker that resumes it is given only what is left.
        self.slept = 0.0

    async def open(self) -> Any:
        from ._runtime import checkout

        self.session = await (await checkout(self._config, self.slept)).__aenter__()
        return self.session

    async def close(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            await session.__aexit__(None, None, None)


class ExternalFunctionBridge:
    """Drives Monty execution with external function dispatch.

    Uses the low-level `session.feed_start()` / `snapshot.resume()` loop,
    awaiting each external call before resuming. Sandbox code therefore calls
    externals synchronously — `add(x, y)` yields the value, no `await` — while
    async external functions (task.aio, durable ops) are awaited on the host.

    With *awaitable_tasks*, a task call instead starts the task and returns a
    future, which the source awaits, alone or with `asyncio.gather`, so several
    tasks run at once. Monty pauses when every sandbox coroutine is waiting on
    one, and the bridge resumes it with each task's result as it finishes. While
    it waits, the paused source is dumped and its worker returned to the pool.
    """

    def __init__(
        self,
        task_refs: Dict[str, Any],
        trace_refs: Dict[str, Any],
        durable_refs: Dict[str, Any],
        awaitable_tasks: bool = False,
    ) -> None:
        self.task_refs = task_refs
        self.trace_refs = trace_refs
        self.durable_refs = durable_refs
        self.awaitable_tasks = awaitable_tasks
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
        """Handle a `flyte_map("task_name", *iterables, **kwargs)` call.

        Resolves the task name to the real `TaskTemplate`, then delegates
        to `flyte.map.aio` so that concurrency, group tracking, and
        `return_exceptions` all work identically to top-level `flyte.map`.
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

        from flyte._task import TaskTemplate

        if isinstance(task, TaskTemplate):
            results: List[Any] = []
            async for r in flyte_map.aio(task, *iterables, **map_kwargs):  # type: ignore[arg-type]
                results.append(r)
            return results

        return await self._map_callable(task, iterables, **map_kwargs)

    async def _map_callable(
        self,
        fn: Callable[..., Any],
        iterables: list[Any],
        *,
        concurrency: int | None = None,
        return_exceptions: bool = False,
        group_name: str | None = None,
    ) -> list[Any]:
        """Map *fn* over zipped *iterables* (for sandbox tools that are not TaskTemplates)."""
        del group_name  # only supported for durable flyte.map over TaskTemplate
        rows = list(zip(*iterables)) if iterables else []

        async def run_row(row: tuple[Any, ...]) -> Any:
            try:
                return await _call_external(fn, *row)
            except Exception as exc:
                if return_exceptions:
                    return exc
                raise

        if concurrency and concurrency > 0:
            sem = asyncio.Semaphore(concurrency)

            async def limited(row: tuple[Any, ...]) -> Any:
                async with sem:
                    return await run_row(row)

            return list(await asyncio.gather(*[limited(row) for row in rows]))

        results: list[Any] = []
        for row in rows:
            results.append(await run_row(row))
        return results

    @staticmethod
    async def _handle_os_call(progress: Any, worker: _Worker) -> Any:
        """Resume a suspended OS call (`progress.is_os_function`).

        With `feed_start` Monty leaves sleeps to the host. It has already
        charged the sleep against `max_total_sleep_secs`, so only the wait
        itself remains. Every other OS call (environment, filesystem, ...) is
        declined, which raises inside the sandbox at the offending line.
        """
        if progress.function_name in _SLEEP_FUNCTIONS:
            worker.slept += progress.args[0]
            await asyncio.sleep(progress.args[0])
            return await progress.resume({"return_value": None})
        return await progress.resume_not_handled()

    @staticmethod
    async def _settle(running: Dict[int, "asyncio.Task[Any]"]) -> Dict[int, Any]:
        """Wait for at least one running task call and return the results of all that are done.

        A call that fails with `RaiseInSandbox` is raised in the sandbox where it is awaited;
        any other failure propagates, failing the run as a failed synchronous call does.
        """
        done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
        results: Dict[int, Any] = {}
        for call_id, task in list(running.items()):
            if task not in done:
                continue
            try:
                results[call_id] = {"return_value": _to_monty(task.result())}
            except RaiseInSandbox as raised:
                results[call_id] = {"exception": raised.exception}
            del running[call_id]
        return results

    async def execute_monty(self, config: Optional["SandboxedConfig"], code: str, inputs: Dict[str, Any]) -> Any:
        """Run *code* on a pooled worker configured from *config*.

        Args:
            config: The sandbox's limits.
            code: The rewritten function body source.
            inputs: Mapping of input parameter names to values.

        An exception raised by an external function propagates out unchanged,
        abandoning the suspended feed; the worker is returned to the pool
        regardless, and task calls still running are cancelled.
        """
        worker = _Worker(config)
        running: Dict[int, asyncio.Task[Any]] = {}
        try:
            await worker.open()
            return await self._drive(worker, code, inputs, running)
        finally:
            for task in running.values():
                if task.done() and not task.cancelled():
                    # Read so asyncio does not log it as never retrieved; the run has failed already.
                    task.exception()
                else:
                    task.cancel()
            await worker.close()

    async def _drive(
        self, worker: _Worker, code: str, inputs: Dict[str, Any], running: Dict[int, "asyncio.Task[Any]"]
    ) -> Any:
        """Feed *code* to *worker*'s session and answer its suspensions until it completes."""
        from pydantic_monty import AsyncFunctionSnapshot, AsyncFutureSnapshot, AsyncNameLookupSnapshot, MontyComplete

        ext_fns = self._build_external_functions()

        # Marshal any IO types in the initial inputs so Monty can hold them
        monty_inputs = {k: _to_monty(v) for k, v in inputs.items()}

        progress = await worker.session.feed_start(code, inputs=monty_inputs)

        while True:
            if isinstance(progress, MontyComplete):
                return _from_monty(progress.output)
            elif isinstance(progress, AsyncFunctionSnapshot):
                if progress.is_os_function:
                    progress = await self._handle_os_call(progress, worker)
                    continue

                # Handle flyte_map as a special built-in for parallel execution
                if progress.function_name == "flyte_map":
                    args = [_from_monty(a) for a in progress.args]
                    kwargs = {k: _from_monty(v) for k, v in progress.kwargs.items()}
                    result = await self._handle_flyte_map(args, kwargs)
                    progress = await progress.resume({"return_value": _to_monty(result)})
                    continue

                fn = ext_fns.get(progress.function_name)
                if fn is None:
                    raise RuntimeError(f"Sandboxed task called unknown external function: {progress.function_name}")

                # Unmarshal IO handles before calling the external function
                args = [_from_monty(a) for a in progress.args]
                kwargs = {k: _from_monty(v) for k, v in progress.kwargs.items()}

                if self.awaitable_tasks and progress.function_name in self.task_refs:
                    running[progress.call_id] = asyncio.create_task(_call_external(fn, *args, **kwargs))
                    progress = await progress.resume({"future": ...})
                    continue

                try:
                    result = await _call_external(fn, *args, **kwargs)
                except RaiseInSandbox as raised:
                    progress = await progress.resume({"exception": raised.exception})
                    continue

                progress = await progress.resume({"return_value": _to_monty(result)})
            elif isinstance(progress, AsyncFutureSnapshot):
                if not running:
                    raise RuntimeError("The sandbox is waiting on a future no task call returned")
                state = progress.dump()
                await worker.close()
                results = await self._settle(running)
                restored = await (await worker.open()).load_snapshot(state)
                progress = await restored.resume(results)
            elif isinstance(progress, AsyncNameLookupSnapshot):
                # A bare reference to an undefined name (externals are only
                # reachable as calls). Leave it unbound so the sandbox raises
                # NameError at the offending line, as an ordinary script would.
                progress = await progress.resume()
            else:
                raise RuntimeError(f"Unexpected Monty progress state: {progress!r}")
