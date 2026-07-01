"""OpenAI Agents SDK adapter for Flyte.

Bring your own ``openai-agents`` ``Agent`` (tools, handoffs, guardrails) and run
it durably on Flyte. The adapter provides three things, each independently
usable:

- :func:`function_tool` — turn a Flyte ``@env.task`` into an OpenAI Agents tool
  that executes as a durable child action (own container/GPU, retries,
  caching) when the agent calls it.
- :class:`FlyteModelProvider` — a ``ModelProvider`` wrapper that records each
  model turn through ``flyte.trace`` so a crashed/retried run replays
  completed turns instead of re-calling (and re-billing) the LLM.
- :class:`FlyteTracingProcessor` — forwards the OpenAI Agents trace (turns, tool
  calls, handoffs, token usage) into the Flyte task report for observability.

:func:`run_agent` wires all three together for the common case. For full control,
use them directly with ``Runner.run`` and a ``RunConfig``.
"""

from ._durable import FlyteModel, FlyteModelProvider
from ._memory import FlyteSession
from ._observability import FlyteTracingProcessor, install_flyte_tracing
from ._run import run_agent
from ._tools import FunctionTool, function_tool

__all__ = [
    "FlyteModel",
    "FlyteModelProvider",
    "FlyteSession",
    "FlyteTracingProcessor",
    "FunctionTool",
    "function_tool",
    "install_flyte_tracing",
    "run_agent",
]
