"""LangChain adapter for Flyte.

Bring your own LangChain agent and run it durably on Flyte. The adapter provides:

- :func:`tool` — turn a Flyte ``@env.task`` into a LangChain ``StructuredTool``
  (a ``BaseTool``) that executes as a durable child action (own container/GPU,
  retries, caching).
- :func:`run_agent` — run the LangChain agent (a compiled ``create_agent`` graph)
  inside your task and return the final answer. Either pass a pre-built ``agent``
  or let it build one from ``tools`` + ``model`` + ``instructions``.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from ._run import run_agent, run_agent_sync
from ._tools import tool

__all__ = ["run_agent", "run_agent_sync", "tool"]
