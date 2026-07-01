# flyteplugins-agents-core

The shared contract every Flyte agent-SDK adapter is built on, so they all
follow one common format.

It is SDK-agnostic — no agent SDK is a dependency here. Adapters depend on it and
implement the contract; `assert_adapter_conforms` enforces that in CI.

## What's in it

- `durable_step(request_key, run, *, name, dumps, loads)` — record a call as a
  durable, replayable `flyte.trace` leaf (the model-turn durability mechanism).
  The real, non-serializable call is captured in the `run` closure, so the trace
  only ever sees the serializable `request_key`.
- `fingerprint(payload)` — a deterministic memo key from a request payload.
- `ToolTaskResolver` / `attach_tool_resolver(task)` — make a tool-backing task
  resolve to itself on the worker (via the tool's `__wrapped_task__` hook)
  instead of re-dispatching, which would otherwise recurse indefinitely.
- `ReportTimeline` — render agent events into a tab of the Flyte task report.
- `flyteplugins.agents.core.testing.assert_adapter_conforms(module)` — the
  conformance check every adapter runs as a one-line test.

## The rule every adapter follows

The agent run is a Flyte `@env.task` (the durable parent); each model turn
is a `flyte.trace` (a memoized, replayable leaf); each tool is a Flyte task
invoked as a durable child action.

## Writing an adapter

A `flyteplugins-agents-<sdk>` package depends on this core and exposes, at
minimum, `function_tool` and `run_agent` (see the conformance contract). Its tool
object exposes `__wrapped_task__`; `run_agent` runs the SDK's loop inside the
calling `@env.task`, wraps the model in a durable provider built on `durable_step`,
and renders the trace via `ReportTimeline`.
