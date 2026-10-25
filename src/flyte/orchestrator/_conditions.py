"""Conditions inside `flyte.orchestrator.task`.

The sandbox can only hold plain values, not a condition object, so an
orchestrator that uses conditions is shipped with a few lines of Python ahead
of its source that define `new_condition` and a condition's `wait()` in terms
of two host calls:

- `_flyte_new_condition(name, **options)` creates the condition and returns a
  handle to it, a string: the name of its child action.
- `_flyte_wait_condition(handle)` waits until it is signalled and returns the
  value it was signalled with.

The worker answers them by creating a condition child action and waiting on
it, with no pod: the orchestrator holds only its paused interpreter until
someone signals the condition from the UI or the API. Run locally,
`LocalConditions` answers them with `flyte.new_condition`.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Mapping, Tuple

NEW_CONDITION = "_flyte_new_condition"
WAIT_CONDITION = "_flyte_wait_condition"

# Monty runs this, so it keeps to what Monty supports. The worker's tests run
# the same text (sandbox-leaseworker, crates/orchestrator-core/tests/conditions.rs).
PRELUDE = """\
class _FlyteCondition:
    def __init__(self, name, handle):
        self.name = name
        self._handle = handle

    def wait(self):
        return _flyte_wait_condition(self._handle)


def _flyte_new_condition_impl(
    name, /, prompt="Approve?", prompt_type="text", data_type=bool, description="", timeout=None
):
    if data_type not in (bool, int, float, str):
        raise TypeError(f"Invalid data_type {data_type}. Must be one of bool, int, float or str.")
    if timeout is not None and not isinstance(timeout, (int, float)):
        timeout = timeout.total_seconds()
    handle = _flyte_new_condition(
        name,
        prompt=prompt,
        prompt_type=prompt_type,
        data_type=data_type.__name__,
        description=description,
        timeout=timeout,
    )
    return _FlyteCondition(name, handle)


class _FlyteNamespace:
    def new_condition(
        self, name, /, prompt="Approve?", prompt_type="text", data_type=bool, description="", timeout=None
    ):
        return _flyte_new_condition_impl(
            name,
            prompt=prompt,
            prompt_type=prompt_type,
            data_type=data_type,
            description=description,
            timeout=timeout,
        )
"""

_DATA_TYPES: Dict[str, type] = {"bool": bool, "int": int, "float": float, "str": str}


def find_condition_refs(func: Callable, trace_refs: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Find how *func* reaches `flyte.new_condition`, and take those names out of *trace_refs*.

    Returns the names bound to `new_condition` itself (`from flyte import new_condition`), and the
    names bound to the `flyte` module when the function uses its `new_condition` (`import flyte`).
    """
    from flyte._condition import new_condition

    functions = sorted(name for name, ref in trace_refs.items() if ref is new_condition)
    for name in functions:
        del trace_refs[name]

    code = getattr(func, "__code__")
    flyte_module = sys.modules["flyte"]
    func_globals = getattr(func, "__globals__", {})
    modules: List[str] = []
    if "new_condition" in code.co_names:
        modules = sorted(
            name for name in set(code.co_names) | set(code.co_freevars) if func_globals.get(name) is flyte_module
        )
    return functions, modules


def prelude(functions: List[str], modules: List[str]) -> str:
    """The prelude, binding each of these names as the source uses it."""
    bindings = [f"{name} = _flyte_new_condition_impl" for name in functions]
    bindings += [f"{name} = _FlyteNamespace()" for name in modules]
    return PRELUDE + "\n\n" + "\n".join(bindings) + "\n\n\n"


class LocalConditions:
    """Answers the prelude's host calls with `flyte.new_condition`, for a run on this machine."""

    def __init__(self) -> None:
        self._conditions: Dict[str, Any] = {}

    def functions(self) -> Mapping[str, Callable[..., Any]]:
        return {NEW_CONDITION: self.new_condition, WAIT_CONDITION: self.wait}

    async def new_condition(self, name: str, /, data_type: str = "bool", **options: Any) -> str:
        import flyte

        condition = await flyte.new_condition.aio(name, data_type=_DATA_TYPES[data_type], **options)
        handle = f"{name}-{len(self._conditions)}"
        self._conditions[handle] = condition
        return handle

    async def wait(self, handle: str) -> Any:
        condition = self._conditions.get(handle)
        if condition is None:
            raise RuntimeError(f"{handle!r} is not a condition this orchestrator created")
        return await condition.wait.aio()
