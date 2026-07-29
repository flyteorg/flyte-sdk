"""Hermes agent adapter for Flyte.

Bring your own Hermes agent (the ``hermes-agent`` package's ``AIAgent``) and
run it durably on Flyte. The adapter provides:

- :func:`tool` — turn a Flyte ``@env.task`` into a Hermes tool that executes as
  a durable child action (own container/GPU, retries, caching), registered in
  the Hermes tool registry under the :data:`FLYTE_TOOLSET` toolset.
- :func:`run_agent` — run the Hermes agent loop inside your task and return the
  final answer.

Each tool call runs as a durable Flyte child action, and the run timeline is
rendered into the Flyte task report.
"""

from ._run import run_agent, run_agent_sync
from ._tools import FLYTE_TOOLSET, tool

__all__ = ["FLYTE_TOOLSET", "run_agent", "run_agent_sync", "tool"]
