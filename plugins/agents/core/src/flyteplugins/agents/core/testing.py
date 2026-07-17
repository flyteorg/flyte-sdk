"""Conformance harness — enforce the common adapter format.

Every ``flyteplugins.agents.<sdk>`` adapter ships a one-line test::

    from flyteplugins.agents.core.testing import assert_adapter_conforms
    import flyteplugins.agents.openai as adapter

    def test_conformance():
        assert_adapter_conforms(adapter)

CI then fails if an adapter drifts from the shared format.
"""

from __future__ import annotations

import inspect
import typing

# The keyword surface every ``run_agent`` must accept, so the call shape is the
# same across SDKs.
REQUIRED_RUN_AGENT_PARAMS = ("tools", "model", "instructions", "durable", "observability", "memory_key")


def assert_adapter_conforms(adapter: typing.Any) -> None:
    """Assert that an adapter module implements the common agent-adapter contract.

    The contract:

    1. exports a callable ``tool`` that turns an ``@env.task`` into the
       SDK's tool type, attaching :class:`~flyteplugins.agents.core.ToolTaskResolver`
       and exposing ``__wrapped_task__`` (so the task does not self-recurse on the
       worker);
    2. exports an async ``run_agent`` (awaited from async tasks) and a plain
       ``run_agent_sync`` companion (called from sync tasks), both accepting the
       standard keyword surface.

    Raises ``AssertionError`` with a specific message on any deviation.
    """
    import flyte

    from flyteplugins.agents.core import ToolTaskResolver

    name = getattr(adapter, "__name__", repr(adapter))

    tool = getattr(adapter, "tool", None)
    assert callable(tool), f"{name}: must export a callable `tool`"

    # ``run_agent`` is a plain coroutine function so it runs on the caller's own
    # event loop; ``run_agent_sync`` is its synchronous companion for sync tasks.
    run_agent = getattr(adapter, "run_agent", None)
    assert callable(run_agent), f"{name}: must export a callable `run_agent`"
    assert inspect.iscoroutinefunction(run_agent), (
        f"{name}: `run_agent` must be an async function (use `run_agent_sync` for the sync path)"
    )
    params = inspect.signature(run_agent).parameters
    for required in REQUIRED_RUN_AGENT_PARAMS:
        assert required in params, f"{name}: `run_agent` must accept `{required}`"

    # The keyword surface is not re-checked here: sync_variant preserves
    # run_agent's signature via functools.wraps, so the check above covers it.
    run_agent_sync = getattr(adapter, "run_agent_sync", None)
    assert callable(run_agent_sync), f"{name}: must export a callable `run_agent_sync`"
    assert not inspect.iscoroutinefunction(run_agent_sync), (
        f"{name}: `run_agent_sync` must be a plain synchronous callable"
    )

    # tool on a task must expose the real task and wire the resolver.
    env = flyte.TaskEnvironment("agents-core-conformance")

    @tool
    @env.task
    def _sample(city: str) -> str:
        """A sample tool task."""
        return city

    wrapped = getattr(_sample, "__wrapped_task__", None)
    assert wrapped is not None, f"{name}: `tool` result must expose `__wrapped_task__`"
    assert isinstance(getattr(wrapped, "task_resolver", None), ToolTaskResolver), (
        f"{name}: the tool's backing task must use ToolTaskResolver (or it self-recurses on the worker)"
    )
