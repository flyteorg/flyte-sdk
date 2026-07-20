"""Deep Agents adapter for Flyte.

Bring your own `Deep Agent <https://docs.langchain.com/oss/python/deepagents/overview>`_
— LangChain's agent harness with built-in planning, a virtual filesystem, and
subagents — and run it durably on Flyte. The adapter provides:

- :func:`tool` — turn a Flyte ``@env.task`` into a LangChain ``StructuredTool``
  (a ``BaseTool``) that executes as a durable child action (own container/GPU,
  retries, caching). Attach it to the main agent or to a subagent.
- :func:`run_agent` — run the deep agent (a compiled ``create_deep_agent``
  graph) inside your task and return the final answer. Either pass a pre-built
  ``agent`` or let it build one from ``tools`` + ``model`` + ``instructions``
  (Deep-Agents options like ``subagents=`` pass through).
- :class:`DurableChatModel` — wrap any LangChain chat model so each model turn
  is recorded/replayed via ``flyte.trace``; use it when building your own agent
  with ``create_deep_agent(model=DurableChatModel(inner=...))``.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report. ``memory_key`` persists the conversation
*and* the agent's virtual filesystem across runs.
"""

from ._durable import DurableChatModel
from ._run import run_agent, run_agent_sync
from ._tools import tool

__all__ = ["DurableChatModel", "run_agent", "run_agent_sync", "tool"]
