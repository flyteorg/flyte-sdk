"""flyteplugins-agents-core — the shared contract every agent-SDK adapter implements.

This package holds the SDK-agnostic machinery that makes Flyte the durable
runtime under any agent framework, so each ``flyteplugins-agents-<sdk>`` adapter
stays thin and consistent:

- :func:`durable_step` — record a call as a durable, replayable ``flyte.trace``
  leaf (the model-turn durability mechanism), keyed by a fingerprint.
- :func:`fingerprint` — a deterministic memo key from a request payload.
- :class:`ToolTaskResolver` / :func:`attach_tool_resolver` — make a tool-backing
  task resolve to itself on the worker.
- :class:`ReportTimeline` — render agent events into the Flyte task report.
- :mod:`flyteplugins.agents.core.testing` — :func:`assert_adapter_conforms`, the
  CI-enforced conformance check every adapter runs.

The division of labor every adapter follows: the agent run is a Flyte
``@env.task`` (durable parent), each model turn is a ``flyte.trace`` (a
memoized, replayable leaf), and each tool is a Flyte task invoked as a
durable child action.
"""

from ._durable import durable_step
from ._fingerprint import fingerprint, jsonable
from ._memory import resolve_memory
from ._observability import ReportTimeline, abbrev, duration_ms, flush_report
from ._sync import run_coro_sync, sync_variant
from ._tools import ToolTaskResolver, attach_tool_resolver, coerce_tool_args, task_json_schema, tool

__all__ = [
    "ReportTimeline",
    "ToolTaskResolver",
    "abbrev",
    "attach_tool_resolver",
    "coerce_tool_args",
    "durable_step",
    "duration_ms",
    "fingerprint",
    "flush_report",
    "jsonable",
    "resolve_memory",
    "run_coro_sync",
    "sync_variant",
    "task_json_schema",
    "tool",
]
