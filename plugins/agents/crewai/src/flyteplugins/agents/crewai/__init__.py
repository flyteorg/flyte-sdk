"""CrewAI adapter for Flyte.

Bring your own CrewAI agent and run it durably on Flyte. The adapter provides:

- :class:`CrewAIAgent` — build a CrewAI agent with durable tools,
  cross-run memory, and observability.
- :func:`tool` — turn a Flyte ``@env.task`` into a CrewAI tool that executes as
  a durable child action (own container/GPU, retries, caching).
- :func:`run_agent` — run the CrewAI agent loop inside your task and return the
  final answer.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from ._flyte_agent import CrewAIAgent
from ._run import run_agent
from ._tools import tool

__all__ = ["CrewAIAgent", "run_agent", "tool"]
