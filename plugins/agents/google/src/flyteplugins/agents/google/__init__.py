"""Google ADK (Agent Development Kit) adapter for Flyte.

Bring your own ``google-adk`` agent and run it durably on Flyte. ADK's ``Runner``
owns the loop; Flyte is the runtime underneath: the tools you expose are Flyte tasks
(durable child actions), each model turn is recorded for replay (``durable=True``),
the run timeline renders into the task report, and ``memory_key`` gives cross-run
conversation memory.

- :func:`tool` — turn an ``@env.task`` into a Google ADK tool.
- :func:`run_agent` — run the ADK agent loop inside your task and return the answer.
- :func:`durable_model` — wrap a model so its turns are durable, for hand-built agent
  trees (e.g. sub-agent transfers) passed to ``run_agent`` via ``agent=``.

Set the model provider's API key in the environment (e.g. ``GOOGLE_API_KEY`` for
Gemini) — wire it as a Flyte secret.
"""

from flyteplugins.agents.core import tool

from ._durable import FlyteLlm, durable_model
from ._run import run_agent, run_agent_sync

__all__ = ["FlyteLlm", "durable_model", "run_agent", "run_agent_sync", "tool"]
