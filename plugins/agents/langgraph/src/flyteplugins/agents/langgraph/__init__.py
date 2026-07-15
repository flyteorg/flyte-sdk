"""LangGraph adapter for Flyte.

Bring your own LangGraph ``StateGraph`` and run it durably on Flyte. You build the
graph; the adapter provides the durable, observable building blocks:

- :func:`tool` — turn a Flyte ``@env.task`` into a LangChain ``StructuredTool``
  (a first-class LangGraph tool) that executes as a durable child action (own
  container/GPU, retries, caching).
- :func:`ai_node` — the model-calling node: binds the tools to your chat model
  and records each model turn durably (replayed on retry).
- :func:`tool_node` — the tool-executing node: runs the model's tool calls as
  durable Flyte child actions.
- :func:`run_agent` — drive a compiled graph (or build a default one from tools)
  inside your task and return the final answer.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from ._nodes import ai_node, tool_node
from ._run import run_agent
from ._tools import tool

__all__ = ["ai_node", "run_agent", "tool", "tool_node"]
