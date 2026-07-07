"""Claude Agent SDK adapter for Flyte.

Bring your own ``claude-agent-sdk`` agent and run it durably on Flyte. Tools you
expose are Flyte tasks (so each tool call is a durable child action with its own
container/resources, retries and caching); the agent loop itself runs in the
Claude Code runtime, and its timeline is rendered into the Flyte task report.

- :func:`tool` — turn an ``@env.task`` into a Claude in-process MCP tool.
- :func:`run_agent` — run the agent loop inside your task and return the answer.

The ``claude-agent-sdk`` wheel bundles the native ``claude`` CLI (no separate Node.js
install needed); set an Anthropic API key in the environment.
"""

from ._run import run_agent
from ._tools import tool

__all__ = ["tool", "run_agent"]
