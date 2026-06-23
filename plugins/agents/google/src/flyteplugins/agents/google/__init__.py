"""Google ADK (Agent Development Kit) adapter for Flyte.

Bring your own ``google-adk`` agent and run it durably on Flyte. ADK's ``Runner``
owns the loop; Flyte is the runtime underneath: the tools you expose are Flyte tasks
(durable child actions), each model turn is recorded for replay (``durable=True``),
the run timeline renders into the task report, and ``memory_key`` gives cross-run
conversation memory.

- :func:`function_tool` — turn an ``@env.task`` into a Google ADK tool.
- :func:`run_agent` — run the ADK agent loop inside your task and return the answer.

Set the model provider's API key in the environment (e.g. ``GOOGLE_API_KEY`` for
Gemini) — wire it as a Flyte secret.
"""

from flyteplugins.agents.core import function_tool

from ._durable import FlyteLlm
from ._run import run_agent

__all__ = ["FlyteLlm", "function_tool", "run_agent"]
