from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict


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

        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor() as pool:

            async def run_in_pool(func):
                return await loop.run_in_executor(pool, func)

            progress = await run_in_pool(partial(monty.start, inputs=inputs))

            while True:
                if isinstance(progress, MontyComplete):
                    return progress.output
                elif isinstance(progress, MontySnapshot):
                    fn = ext_fns.get(progress.function_name)
                    if fn is None:
                        raise RuntimeError(
                            f"Sandboxed task called unknown external function: {progress.function_name}"
                        )

                    # Call the external function and await if async
                    result = fn(*progress.args, **progress.kwargs)
                    if inspect.iscoroutine(result):
                        result = await result

                    progress = await run_in_pool(partial(progress.resume, return_value=result))
                else:
                    raise RuntimeError(f"Unexpected Monty progress state: {progress!r}")
