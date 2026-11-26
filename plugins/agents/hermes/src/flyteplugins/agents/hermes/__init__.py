"""Hermes agent adapter for Flyte.

Bring your own Hermes agent (the `hermes-agent` package's `AIAgent`) and
run it durably on Flyte. The adapter provides:

- `flyteplugins.agents.hermes.tool` — turn a Flyte `@env.task` into a Hermes tool that executes as
  a durable child action (own container/GPU, retries, caching), registered in
  the Hermes tool registry under the `flyteplugins.agents.hermes.FLYTE_TOOLSET` toolset.
- `flyteplugins.agents.hermes.run_agent` — run the Hermes agent loop inside your task and return the
  final answer.

Each tool call runs as a durable Flyte child action, each model turn is
recorded for replay via `flyte.trace` (through Hermes's `llm_execution`
middleware), and the run timeline is rendered into the Flyte task report.

hermes-agent is not installed with this package. Upstream no longer publishes
to PyPI, and the last release there (0.19.0) has open security advisories.
Install it from git on Python 3.14:

    echo pillow==13.0.0 > overrides.txt
    HERMES_NIX_BUILD=1 uv pip install --override overrides.txt \\
        "hermes-agent @ git+https://github.com/NousResearch/hermes-agent@v0.23.4"

HERMES_NIX_BUILD=1 lets its setup.py build a wheel, and the override settles a
Pillow pin conflict between hermes-agent and browser-harness, the same way
upstream's own lock does.

Recent hermes-agent releases hide plugin tools, Flyte tools included, behind a
tool_search / tool_call bridge. Calls through the bridge still run as durable
Flyte tasks. To show the tools to the model directly, set
tools.tool_search.enabled to off in your hermes config.
"""

from ._run import run_agent, run_agent_sync
from ._tools import FLYTE_TOOLSET, tool

__all__ = ["FLYTE_TOOLSET", "run_agent", "run_agent_sync", "tool"]
