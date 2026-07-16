"""Pydantic AI adapter for Flyte.

Bring your own Pydantic AI ``Agent`` and run it durably on Flyte. The adapter
provides:

- :func:`tool` — turn a Flyte ``@env.task`` into a Pydantic AI tool that
  executes as a durable child action (own container/GPU, retries, caching).
  This is the shared :func:`flyteplugins.agents.core.tool`: Pydantic AI accepts
  plain (async) callables in ``Agent(tools=[...])`` and infers each tool's
  schema from the callable's signature, which the core wrapper preserves.
- :func:`run_agent` — run the Pydantic AI agent loop inside your task and return
  the final answer.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from flyteplugins.agents.core import tool

from ._durable import FlyteModel
from ._run import run_agent

__all__ = ["FlyteModel", "run_agent", "tool"]
