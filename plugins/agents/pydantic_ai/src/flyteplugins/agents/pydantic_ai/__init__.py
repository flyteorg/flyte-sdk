"""Pydantic AI adapter for Flyte.

Bring your own Pydantic AI ``Agent`` and run it durably on Flyte. The adapter
provides:

- :class:`Agent` — build a Pydantic AI ``Agent`` with durable tools,
  cross-run memory, and observability.
- :func:`tool` — turn a Flyte ``@env.task`` into a Pydantic AI ``Tool`` that
  executes as a durable child action (own container/GPU, retries, caching).
- :func:`run_agent` — run the Pydantic AI agent loop inside your task and return
  the final answer.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from ._run import run_agent
from ._tools import tool

__all__ = ["run_agent", "tool"]
