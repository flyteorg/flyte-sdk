"""Mistral Agents adapter for Flyte (mistralai 2.x).

Bring your own Mistral agent and run it durably on Flyte. Tools you
expose are Flyte tasks (each call a durable child action), and each model turn is
recorded via ``flyte.trace`` (the turns are in-process HTTP calls, so we can trace
the seam below the SDK's loop) for per-turn replay on retry.

- :func:`tool` — turn an ``@env.task`` into a Mistral run-framework tool.
- :func:`run_agent` — run the SDK's agent loop inside your task; return the answer.
"""

from flyteplugins.agents.core import tool

from ._run import run_agent

__all__ = ["tool", "run_agent"]
