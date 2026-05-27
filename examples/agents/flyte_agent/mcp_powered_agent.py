"""Agent + remote MCP servers (Slack / GitHub / Linear style).

This example shows how to point a :class:`flyte.ai.agents.Agent` at one
or more **remote MCP servers** so it can transparently use their tools without
any local glue code. We mix:

- a local durable Flyte task as a tool (heavy compute / IO),
- remote MCP tools (e.g. a Slack MCP for posting messages, a GitHub MCP for
  PR comments).

The harness:

1. Connects to each MCP server on first :meth:`Agent.run` call.
2. Lists their tools and registers them under ``<prefix><name>``.
3. From the LLM's perspective these are first-class tools in the catalog.

Requirements::

    pip install 'flyte[mcp]'

Pass ``--prompt "..."`` to drive the agent on the command line.
"""

from __future__ import annotations

import argparse
import os

import flyte
from flyte.ai.agents import Agent, MCPServerSpec

env = flyte.TaskEnvironment(
    name="mcp-agent-tools",
    image=(flyte.Image.from_debian_base().with_pip_packages("litellm", "mcp")),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def compute_release_score(changes: list[str]) -> float:
    """Estimate a release-risk score from a list of recent code-change summaries.

    Replace with your real risk model — this stub keeps the example
    self-contained.
    """
    score = 0.0
    for change in changes:
        text = change.lower()
        if "refactor" in text:
            score += 1.5
        if "schema" in text or "migration" in text:
            score += 3.0
        if "fix" in text:
            score += 0.5
        if "test" in text:
            score -= 0.5
    return round(score, 2)


# ---------------------------------------------------------------------------
# External MCP servers
# ---------------------------------------------------------------------------
#
# Use environment variables so the same example can target real endpoints in
# CI without hard-coding URLs. Each ``MCPServerSpec`` only requires either a
# ``url`` (for HTTP/streamable-http transports) or a ``command`` (for stdio).

mcp_servers: list[MCPServerSpec] = []
if os.getenv("SLACK_MCP_URL"):
    mcp_servers.append(
        MCPServerSpec(
            name="slack",
            url=os.environ["SLACK_MCP_URL"],
            headers={"Authorization": f"Bearer {os.environ.get('SLACK_MCP_TOKEN', '')}"},
            tool_prefix="slack_",
        ),
    )
if os.getenv("GITHUB_MCP_URL"):
    mcp_servers.append(
        MCPServerSpec(
            name="github",
            url=os.environ["GITHUB_MCP_URL"],
            headers={"Authorization": f"Bearer {os.environ.get('GITHUB_MCP_TOKEN', '')}"},
            tool_prefix="gh_",
            # Only expose a narrow tool surface to the LLM.
            tool_filter=["list_pull_requests", "comment_on_pull_request"],
        ),
    )


agent = Agent(
    name="release-shepherd",
    instructions=(
        "You are a release shepherd. For each request:\n"
        "1. Use the GitHub MCP tools to inspect recent PRs.\n"
        "2. Use compute_release_score to quantify the risk.\n"
        "3. Use the Slack MCP tools to post a structured summary to the "
        "release channel.\n"
        "Be precise and avoid unnecessary chatter."
    ),
    model="claude-haiku-4-5",
    tools=[compute_release_score],
    mcp_servers=mcp_servers,
    max_turns=18,
)


@env.task(report=True)
async def shepherd(prompt: str) -> str:
    """Run the release shepherd agent for a one-off task."""
    result = await agent.run(prompt)
    return result.summary or result.error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP-powered Agent.")
    parser.add_argument(
        "--prompt",
        default=(
            "Look at the last 10 merged PRs in our main branch, score the release risk, and post a digest to #releases."
        ),
    )
    args = parser.parse_args()

    flyte.init_from_config()
    run = flyte.run(shepherd, prompt=args.prompt)
    print(f"Run URL: {run.url}")
    run.wait()
    print(run.outputs())
