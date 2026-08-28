"""DeepSeek Harness adapter for Flyte.

Bring your own `deepseek-harness-sdk` agent and run it durably on Flyte. The
harness owns the agent loop (it drives the model inside its own runtime
subprocess); Flyte is the durable runtime underneath — retries / self-healing,
per-tool containerized execution, cross-run memory, and the agent timeline in
the task report.

- `flyteplugins.agents.deepseek.tool` — turn an `@env.task` into a harness tool.
- `flyteplugins.agents.deepseek.run_agent` — run the agent loop inside your task
  and return the answer.

Tools work differently here than in the client-side SDKs. The harness has no
tool-registration message: its tool surface is whatever its Cordis composition
provides, inside the runtime subprocess. So a Flyte-task tool is published into
the harness workspace as an executable shim that calls back into this process,
where the real task runs as a durable child action. The mechanism is in
`._bridge`; the call shape is the shared one every adapter presents.

The `deepseek-harness-sdk` wheel pulls in the bundled `deepseek-harness-runtime-bin`
runtime (no separate Node.js install needed); set `DEEPSEEK_API_KEY` in the
environment, and `DEEPSEEK_BASE_URL` if you route through a proxy.
"""

from ._bridge import TOOLS_DIRNAME, ToolBridge
from ._run import run_agent, run_agent_sync
from ._tools import HarnessTool, tool

__all__ = ["TOOLS_DIRNAME", "HarnessTool", "ToolBridge", "run_agent", "run_agent_sync", "tool"]
