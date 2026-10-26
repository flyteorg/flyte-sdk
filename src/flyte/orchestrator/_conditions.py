"""Conditions inside `flyte.orchestrator.task`.

The sandbox can only hold plain values, not a condition object, so an
orchestrator that uses conditions is shipped with a few lines of Python ahead
of its source that define `new_condition` and a condition's `wait()` in terms
of two host calls:

- `_flyte_new_condition(name, **options)` creates the condition and returns a
  handle to it, a string: the name of its child action.
- `_flyte_wait_condition(handle, name=, data_type=)` waits until it is
  signalled and returns the value it was signalled with, as `data_type`.

The worker answers them by creating a condition child action and waiting on
it, with no pod: the orchestrator holds only its paused interpreter until
someone signals the condition from the UI or the API. Run locally,
`LocalConditions` answers them with `flyte.new_condition`.

A condition that does not end with a value raises `RuntimeError("<code>:
<message>")` from `_flyte_wait_condition`, the code being the name of the
`flyte.errors` class the SDK raises. The sandbox has no exception classes of
its own, so the prelude stands a builtin in for each of those:
`ConditionTimedoutError` is `TimeoutError`, so that catching it catches only
timeouts, and `ConditionFailedError` and `ActionAbortedError` are
`RuntimeError`.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Mapping

NEW_CONDITION = "_flyte_new_condition"
WAIT_CONDITION = "_flyte_wait_condition"

# The `flyte.errors` classes a condition's wait raises, and what stands for each in the sandbox.
ERRORS = {
    "ConditionTimedoutError": "TimeoutError",
    "ConditionFailedError": "RuntimeError",
    "ActionAbortedError": "RuntimeError",
}

# Monty runs this, so it keeps to what Monty supports: no class inheritance, and
# class bodies of methods and simple class variables. The worker's tests run the
# same text (sandbox-leaseworker, crates/orchestrator-core/tests/conditions.rs).
PRELUDE = """\
class _FlyteErrors:
    ConditionTimedoutError = TimeoutError
    ConditionFailedError = RuntimeError
    ActionAbortedError = RuntimeError


class _FlyteCondition:
    def __init__(self, name, handle, data_type):
        self.name = name
        self._handle = handle
        self._data_type = data_type

    def wait(self):
        try:
            return _flyte_wait_condition(self._handle, name=self.name, data_type=self._data_type)
        except RuntimeError as error:
            if str(error).startswith("ConditionTimedoutError: "):
                raise TimeoutError(str(error)) from None
            raise


def _flyte_new_condition_impl(
    name, /, prompt="Approve?", prompt_type="text", data_type=bool, description="", timeout=None
):
    if data_type not in (bool, int, float, str):
        raise TypeError(f"Invalid data_type {data_type}. Must be one of bool, int, float or str.")
    if prompt_type not in ("text", "markdown"):
        raise ValueError(f"prompt_type must be 'text' or 'markdown', not {prompt_type!r}")
    if timeout is not None:
        if not isinstance(timeout, (int, float)):
            timeout = timeout.total_seconds()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
    handle = _flyte_new_condition(
        name,
        prompt=prompt,
        prompt_type=prompt_type,
        data_type=data_type.__name__,
        description=description,
        timeout=timeout,
    )
    return _FlyteCondition(name, handle, data_type.__name__)


class _FlyteNamespace:
    errors = _FlyteErrors

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


def find_condition_refs(func: Callable, trace_refs: Dict[str, Any]) -> Dict[str, str]:
    """Find the globals through which *func* reaches conditions, and take them out of *trace_refs*.

    Returns what the prelude binds each of them to: `new_condition` itself (`from flyte import
    new_condition`), the `flyte` module when the function uses its `new_condition` or `errors`
    (`import flyte`), and `flyte.errors` or the condition errors in it (`from flyte.errors import
    ConditionTimedoutError`). Empty if the function uses none of them.
    """
    import flyte.errors
    from flyte._condition import new_condition

    bindings: Dict[str, str] = {}
    for name in [name for name, ref in trace_refs.items() if ref is new_condition]:
        del trace_refs[name]
        bindings[name] = "_flyte_new_condition_impl"

    code = getattr(func, "__code__")
    func_globals = getattr(func, "__globals__", {})
    flyte_module = sys.modules["flyte"]
    errors = {getattr(flyte.errors, error): error for error in ERRORS}
    uses_flyte = "new_condition" in code.co_names or "errors" in code.co_names
    for name in set(code.co_names) | set(code.co_freevars):
        ref = func_globals.get(name)
        if ref is flyte_module and uses_flyte:
            bindings[name] = "_FlyteNamespace()"
        elif ref is flyte.errors:
            bindings[name] = "_FlyteErrors"
        elif isinstance(ref, type) and ref in errors:
            bindings[name] = f"_FlyteErrors.{errors[ref]}"
    # The prelude only defines errors for conditions to raise.
    if not any(value in ("_flyte_new_condition_impl", "_FlyteNamespace()") for value in bindings.values()):
        return {}
    return bindings


def prelude(bindings: Mapping[str, str]) -> str:
    """The prelude, binding each name as `find_condition_refs` found it."""
    lines = [f"{name} = {bindings[name]}" for name in sorted(bindings)]
    return PRELUDE + "\n\n" + "\n".join(lines) + "\n\n\n"


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

    async def wait(self, handle: str, /, name: str = "", data_type: str = "bool") -> Any:
        import flyte.errors
        from flyte.sandbox._bridge import RaiseInSandbox

        condition = self._conditions.get(handle)
        if condition is None:
            raise RuntimeError(f"{handle!r} is not a condition this orchestrator created")
        try:
            return await condition.wait.aio()
        except tuple(getattr(flyte.errors, error) for error in ERRORS) as error:
            # Raised in the source as the worker raises it, where it can be caught.
            raise RaiseInSandbox(RuntimeError(f"{type(error).__name__}: {error}")) from error
