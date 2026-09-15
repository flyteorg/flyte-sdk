from __future__ import annotations

import contextvars
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_current_dbt_node_status: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_dbt_node_status",
    default="",
)

_TRACED_DBT_NODE_EVENTS = {"NodeFinished"}

DbtEventCallback = Callable[[Any], None]


@dataclass
class DbtNodeResult:
    """Small serializable summary of one dbt node result."""

    unique_id: str
    name: str
    resource_type: str
    status: str
    message: Optional[str] = None
    failures: Optional[int] = None
    execution_time: Optional[float] = None
    relation_name: Optional[str] = None


def _node_name(node: Any) -> str:
    return str(getattr(node, "name", "") or getattr(node, "unique_id", ""))


def _node_resource_type(node: Any) -> str:
    resource_type = getattr(node, "resource_type", "")
    if hasattr(resource_type, "value"):
        return str(resource_type.value)
    return str(resource_type)


def _summarize_node_result(result: Any) -> DbtNodeResult:
    node = getattr(result, "node", None)
    return DbtNodeResult(
        unique_id=str(getattr(result, "unique_id", "") or getattr(node, "unique_id", "")),
        name=_node_name(node),
        resource_type=_node_resource_type(node),
        status=str(getattr(result, "status", "")),
        message=getattr(result, "message", None),
        failures=getattr(result, "failures", None),
        execution_time=getattr(result, "execution_time", None),
        relation_name=getattr(result, "relation_name", None),
    )


def _raw_node_results(runner_result: Any) -> list[Any]:
    raw_results = getattr(runner_result, "result", None)
    if raw_results is None:
        return []

    if hasattr(raw_results, "results"):
        raw_results = raw_results.results

    if isinstance(raw_results, tuple):
        raw_results = list(raw_results)
    elif not isinstance(raw_results, list):
        raw_results = []

    return raw_results


def summarize_dbt_runner_result(runner_result: Any) -> list[DbtNodeResult]:
    return [_summarize_node_result(result) for result in _raw_node_results(runner_result)]


def _stringify_status(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    return str(status)


def _node_info_value(node_info: Any, key: str) -> Any:
    if isinstance(node_info, dict):
        return node_info.get(key)
    return getattr(node_info, key, None)


def _event_name(event: Any) -> Optional[str]:
    info = getattr(event, "info", None)
    name = getattr(info, "name", None)
    if name:
        return str(name)

    name = getattr(event, "name", None)
    if name:
        return str(name)

    return type(event).__name__


def _traced_dbt_node_status(node_name: str, trace_name: str) -> str:
    import flyte

    def dbt_node_status(node_name: str) -> str:
        return _current_dbt_node_status.get()

    dbt_node_status.__name__ = trace_name
    dbt_node_status.__qualname__ = trace_name

    return flyte.trace(dbt_node_status)(node_name)


def _record_dbt_node_status(
    node_name: str,
    node_status: str,
    trace_name: Optional[str] = None,
) -> str:
    token = _current_dbt_node_status.set(node_status)
    try:
        return _traced_dbt_node_status(
            node_name,
            trace_name=trace_name or node_name,
        )
    finally:
        _current_dbt_node_status.reset(token)


def on_event(event: Any) -> None:
    """Default dbt event callback that records dbt node status as a Flyte trace."""
    event_name = _event_name(event)
    if event_name not in _TRACED_DBT_NODE_EVENTS:
        return

    data = getattr(event, "data", None)
    if data is None:
        return

    node_info = getattr(data, "node_info", None)
    if node_info is None:
        return

    node_name = _node_info_value(node_info, "node_name") or _node_info_value(node_info, "unique_id")
    if not node_name:
        return

    status = _node_info_value(node_info, "node_status") or getattr(data, "status", None)
    if status is None:
        return

    _record_dbt_node_status(
        str(node_name),
        _stringify_status(status),
        trace_name=f"{status}.{node_name}",
    )


def callback_import_path(callback: DbtEventCallback) -> str:
    module = getattr(callback, "__module__", None)
    name = getattr(callback, "__name__", None)
    qualname = getattr(callback, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or name == "<lambda>":
        raise ValueError("dbt callbacks used in DbtTask must be importable functions when running remotely.")
    return f"{module}.{qualname}"


def import_callback(import_path: str) -> DbtEventCallback:
    module_name, _, attr_path = import_path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid dbt callback import path: {import_path!r}")

    value: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    if not callable(value):
        raise TypeError(f"dbt callback import path does not resolve to a callable: {import_path!r}")
    return value


def resolve_callbacks(callbacks: Sequence[DbtEventCallback | str] | None) -> list[DbtEventCallback]:
    resolved = []
    for callback in callbacks or []:
        if isinstance(callback, str):
            resolved.append(import_callback(callback))
        else:
            resolved.append(callback)
    return resolved


def callback_import_paths(callbacks: Sequence[DbtEventCallback | str] | None) -> list[str]:
    paths = []
    for callback in callbacks or []:
        if isinstance(callback, str):
            paths.append(callback)
        else:
            paths.append(callback_import_path(callback))
    return paths


def _make_on_event_callback(callback: DbtEventCallback | None = None) -> Any:
    from flyte._context import Context, internal_ctx

    parent_context_data = internal_ctx().data

    def wrapped_callback(event: Any) -> None:
        # dbt emits node events from its own worker threads. Re-enter the Flyte
        # task context in those threads so @flyte.trace records remotely.
        with Context(parent_context_data):
            (callback or on_event)(event)

    return wrapped_callback


def invoke_dbt(
    cli_args: list[str],
    callbacks: Sequence[DbtEventCallback | str] | None = None,
    *,
    include_default_callback: bool = True,
) -> list[DbtNodeResult]:
    """Run exactly one dbtRunner invocation and return a serializable summary."""
    from dbt.cli.main import dbtRunner

    args = list(cli_args)
    event_callbacks = []
    if include_default_callback:
        event_callbacks.append(_make_on_event_callback())
    event_callbacks.extend(_make_on_event_callback(callback) for callback in resolve_callbacks(callbacks))

    runner_result = dbtRunner(callbacks=event_callbacks).invoke(args)
    if not runner_result.success:
        if runner_result.exception is not None:
            raise runner_result.exception
        raise RuntimeError(f"dbt invocation failed for cli_args={args!r}")
    return summarize_dbt_runner_result(runner_result)
