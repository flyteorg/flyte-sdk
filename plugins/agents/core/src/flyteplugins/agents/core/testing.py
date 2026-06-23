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

    1. exports a callable ``function_tool`` that turns an ``@env.task`` into the
       SDK's tool type, attaching :class:`~flyteplugins.agents.core.ToolTaskResolver`
       and exposing ``__wrapped_task__`` (so the task does not self-recurse on the
       worker);
    2. exports an async ``run_agent`` accepting the standard keyword surface.

    Raises ``AssertionError`` with a specific message on any deviation.
    """
    import flyte

    from flyteplugins.agents.core import ToolTaskResolver

    name = getattr(adapter, "__name__", repr(adapter))

    function_tool = getattr(adapter, "function_tool", None)
    assert callable(function_tool), f"{name}: must export a callable `function_tool`"

    run_agent = getattr(adapter, "run_agent", None)
    assert inspect.iscoroutinefunction(run_agent), f"{name}: must export an async `run_agent`"
    params = inspect.signature(run_agent).parameters
    for required in REQUIRED_RUN_AGENT_PARAMS:
        assert required in params, f"{name}: `run_agent` must accept `{required}`"

    # function_tool on a task must expose the real task and wire the resolver.
    env = flyte.TaskEnvironment("agents-core-conformance")

    @function_tool
    @env.task
    def _sample(city: str) -> str:
        """A sample tool task."""
        return city

    wrapped = getattr(_sample, "__wrapped_task__", None)
    assert wrapped is not None, f"{name}: `function_tool` result must expose `__wrapped_task__`"
    assert isinstance(getattr(wrapped, "task_resolver", None), ToolTaskResolver), (
        f"{name}: the tool's backing task must use ToolTaskResolver (or it self-recurses on the worker)"
    )
