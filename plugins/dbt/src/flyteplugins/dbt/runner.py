from __future__ import annotations

import importlib
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_MAX_DBT_INVOCATION_ERROR_RESULTS = 10

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


class DbtInvocationError(RuntimeError):
    """Raised when dbt finishes cleanly but reports failed node results."""

    def __init__(self, cli_args: list[str], results: list[DbtNodeResult]):
        self.cli_args = cli_args
        self.results = results
        super().__init__(_format_dbt_invocation_error(cli_args, results))


def _format_dbt_invocation_error(cli_args: list[str], results: list[DbtNodeResult]) -> str:
    if not results:
        return f"dbt invocation failed for cli_args={cli_args!r}"

    failed_results = [result for result in results if result.status.lower() not in {"pass", "success"}]
    if not failed_results:
        failed_results = results

    node_summaries = []
    for result in failed_results[:_MAX_DBT_INVOCATION_ERROR_RESULTS]:
        details = [f"status={result.status!r}"]
        if result.failures is not None:
            details.append(f"failures={result.failures}")
        if result.message:
            details.append(f"message={result.message!r}")
        node_summaries.append(f"{result.unique_id or result.name} ({', '.join(details)})")

    suffix = ""
    if len(failed_results) > _MAX_DBT_INVOCATION_ERROR_RESULTS:
        suffix = f"; and {len(failed_results) - _MAX_DBT_INVOCATION_ERROR_RESULTS} more"

    return f"dbt invocation failed for cli_args={cli_args!r}: {', '.join(node_summaries)}{suffix}"


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


def _is_node_result(result: Any) -> bool:
    return hasattr(result, "node") or hasattr(result, "unique_id") or hasattr(result, "status")


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
    return [_summarize_node_result(result) for result in _raw_node_results(runner_result) if _is_node_result(result)]


def callback_import_path(callback: DbtEventCallback, source_dir: pathlib.Path | None = None) -> str:
    name = getattr(callback, "__name__", None)
    qualname = getattr(callback, "__qualname__", None)
    if not qualname or "<locals>" in qualname or name == "<lambda>":
        raise ValueError("dbt callbacks used in DbtTask must be importable functions when running remotely.")

    if source_dir is None:
        module = getattr(callback, "__module__", None)
        if not module:
            raise ValueError("dbt callbacks used in DbtTask must be importable functions when running remotely.")
    else:
        from flyte._module import extract_obj_module

        module, _ = extract_obj_module(callback, source_dir=source_dir)

    import_path = f"{module}:{qualname}"
    try:
        resolved_callback = import_callback(import_path)
    except (AttributeError, ModuleNotFoundError):
        if source_dir is not None:
            return import_path
        raise
    if resolved_callback is not callback:
        raise ValueError(f"dbt callback {import_path!r} must resolve to the original callback object when imported.")
    return import_path


def _import_dotted_path(import_path: str) -> Any:
    parts = import_path.split(".")
    for module_end in range(len(parts), 0, -1):
        module_name = ".".join(parts[:module_end])
        try:
            value: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attr in parts[module_end:]:
            value = getattr(value, attr)
        return value
    raise ModuleNotFoundError(f"No module found in dbt callback import path: {import_path!r}")


def import_callback(import_path: str) -> DbtEventCallback:
    if ":" in import_path:
        module_name, attr_path = import_path.split(":", 1)
        if not module_name or not attr_path:
            raise ValueError(f"Invalid dbt callback import path: {import_path!r}")

        value: Any = importlib.import_module(module_name)
        for attr in attr_path.split("."):
            value = getattr(value, attr)
    else:
        value = _import_dotted_path(import_path)

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


def callback_import_paths(
    callbacks: Sequence[DbtEventCallback | str] | None,
    source_dir: pathlib.Path | None = None,
) -> list[str]:
    paths = []
    for callback in callbacks or []:
        if isinstance(callback, str):
            import_callback(callback)
            paths.append(callback)
        else:
            paths.append(callback_import_path(callback, source_dir=source_dir))
    return paths


def invoke_dbt(
    cli_args: list[str],
    callbacks: Sequence[DbtEventCallback | str] | None = None,
) -> list[DbtNodeResult]:
    """Run exactly one dbtRunner invocation and return a serializable summary."""
    from dbt.cli.main import dbtRunner

    args = list(cli_args)
    event_callbacks = resolve_callbacks(callbacks)

    runner_result = dbtRunner(callbacks=event_callbacks).invoke(args)
    results = summarize_dbt_runner_result(runner_result)
    if not runner_result.success:
        if runner_result.exception is not None:
            raise runner_result.exception
        raise DbtInvocationError(args, results)
    return results
