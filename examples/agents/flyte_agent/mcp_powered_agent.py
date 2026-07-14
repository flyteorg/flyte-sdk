"""Agent + MCP servers, with the MCP servers deployed as Flyte apps.

This example shows how to point a :class:`flyte.ai.agents.Agent` at one or more
**MCP servers** so it can transparently use their tools without any local glue
code. Instead of depending on external, pre-existing MCP endpoints, it stubs out
a Slack-style and a GitHub-style MCP server using
:class:`flyte.ai.mcp.MCPAppEnvironment` and serves them on the cluster, then
wires the agent to their live endpoints.

We mix:

- a local durable Flyte task as a tool (heavy compute / IO),
- remote MCP tools served by our own ``MCPAppEnvironment`` apps (a Slack stub for
  posting messages, a GitHub stub for inspecting / commenting on PRs).

The harness, on the first :meth:`Agent.run` call:

1. Connects to each MCP server.
2. Lists their tools and registers them under ``<prefix><name>``.
3. From the LLM's perspective these are first-class tools in the catalog.

Requirements::

    pip install 'flyte[mcp]'

Pass ``--prompt "..."`` to drive the agent on the command line.
"""

from __future__ import annotations

import argparse

import flyte
from flyte.ai.agents import Agent, MCPServerSpec
from flyte.ai.mcp import MCPAppEnvironment

# ---------------------------------------------------------------------------
# Stub MCP servers (served as Flyte apps via MCPAppEnvironment)
# ---------------------------------------------------------------------------
#
# These build FastMCP instances with a few canned tools so the example is fully
# self-contained — no external Slack / GitHub MCP endpoints required. Swap the
# stub bodies for real integrations when you go to production.
#
# NOTE: these helpers (and the MCPAppEnvironment construction in ``__main__``)
# import ``mcp.server.fastmcp`` / ``starlette`` / ``uvicorn``, which only the
# *app* images carry. They are intentionally kept out of this module's top level
# so importing it inside the lightweight agent-task container stays cheap.


def _new_fastmcp(name: str):
    """Build a FastMCP configured for serving behind the Flyte apps gateway.

    Two settings matter for a proxied streamable-HTTP deployment:

    - ``stateless_http`` / ``json_response``: serve each call as a self-contained
      JSON response rather than a long-lived SSE session.
    - ``transport_security`` with DNS-rebinding protection disabled: FastMCP
      otherwise validates the request ``Host`` / ``Origin`` against the server's
      own host, which never matches the external gateway host and surfaces as a
      ``421 Misdirected Request``.
    """
    from mcp.server.fastmcp import FastMCP

    transport_security = None
    try:
        from mcp.server.transport_security import TransportSecuritySettings

        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    except Exception:
        transport_security = None

    return FastMCP(
        name=name,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security,
    )


def _build_slack_mcp():
    """A Slack-style MCP server: list channels and post messages (stubs)."""
    mcp = _new_fastmcp("slack-stub")

    @mcp.tool()
    def list_channels() -> list[str]:
        """List the Slack channels the bot can post to."""
        return ["#releases", "#engineering", "#general"]

    @mcp.tool()
    def post_message(channel: str, text: str) -> str:
        """Post a message to a Slack channel. Returns a confirmation string."""
        return f"posted to {channel}: {text}"

    return mcp


def _build_github_mcp():
    """A GitHub-style MCP server: inspect and comment on PRs (stubs)."""
    mcp = _new_fastmcp("github-stub")

    @mcp.tool()
    def list_pull_requests(repo: str, state: str = "merged", limit: int = 10) -> list[dict]:
        """List recent pull requests for ``repo`` (stub data)."""
        canned = [
            {"number": 412, "title": "Refactor scheduler core", "merged": True},
            {"number": 411, "title": "Add migration for events table", "merged": True},
            {"number": 409, "title": "Fix flaky retry test", "merged": True},
            {"number": 407, "title": "Add tests for tool resolver", "merged": True},
        ]
        return canned[:limit]

    @mcp.tool()
    def comment_on_pull_request(repo: str, number: int, body: str) -> str:
        """Post a review comment on a pull request. Returns a confirmation string."""
        return f"commented on {repo}#{number}"

    return mcp


# ``requires_auth=False`` keeps the stub apps reachable from the agent task
# without extra token plumbing. Front real MCP servers with auth in prod.
slack_mcp_env = MCPAppEnvironment(
    name="slack-mcp-stub",
    mcp=_build_slack_mcp(),
    transport="streamable-http",
    requires_auth=False,
)
github_mcp_env = MCPAppEnvironment(
    name="github-mcp-stub",
    mcp=_build_github_mcp(),
    transport="streamable-http",
    requires_auth=False,
)


agent_env = flyte.TaskEnvironment(
    name="mcp-agent-tools",
    image=(flyte.Image.from_debian_base().with_pip_packages("litellm", "mcp")),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    depends_on=[slack_mcp_env, github_mcp_env],
)


@agent_env.task
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


agent_instructions = (
    "You are a release shepherd. For each request:\n"
    "1. Use the GitHub MCP tools to inspect recent PRs.\n"
    "2. Use compute_release_score to quantify the risk.\n"
    "3. Use the Slack MCP tools to post a structured summary to the "
    "release channel.\n"
    "Be precise and avoid unnecessary chatter."
)


@agent_env.task(report=True)
async def shepherd(prompt: str, slack_mcp_url: str, github_mcp_url: str) -> str:
    """Run the release shepherd agent against the deployed stub MCP servers.

    Args:
        prompt: The instruction to drive the agent.
        slack_mcp_url: Streamable-HTTP MCP session URL for the Slack stub app.
        github_mcp_url: Streamable-HTTP MCP session URL for the GitHub stub app.
    """
    mcp_servers = [
        MCPServerSpec(
            name="slack",
            url=slack_mcp_url,
            transport="streamable-http",
            tool_prefix="slack_",
        ),
        MCPServerSpec(
            name="github",
            url=github_mcp_url,
            transport="streamable-http",
            tool_prefix="gh_",
            # Only expose a narrow tool surface to the LLM.
            tool_filter=["list_pull_requests", "comment_on_pull_request"],
        ),
    ]

    agent = Agent(
        name="release-shepherd",
        instructions=agent_instructions,
        model="claude-haiku-4-5",
        tools=[compute_release_score],
        mcp_servers=mcp_servers,
        max_turns=18,
    )
    result = await agent.run.aio(prompt)
    return result.summary or result.error


def _mcp_session_url(endpoint: str) -> str:
    """Streamable-HTTP MCP session URL for an MCPAppEnvironment.

    With ``mcp_mount_path="/mcp"`` and ``transport="streamable-http"`` the
    FastMCP session lives at ``{endpoint}/mcp/mcp`` (mount path + FastMCP's own
    streamable path).
    """
    return f"{endpoint.rstrip('/')}/mcp/mcp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP-powered Agent.")
    parser.add_argument(
        "--prompt",
        default=(
            "Look at the last 10 merged PRs in our main branch on repo "
            "flyteorg/flyte, score the release risk, and post a digest to #releases."
        ),
    )
    args = parser.parse_args()

    from flyte.app._deploy import DeployedAppEnvironment

    flyte.init_from_config()

    deployments = flyte.deploy(agent_env)
    print(f"Agent environment deployed: {deployments[0].summary_repr()}")

    slack_env = deployments[0].envs["slack-mcp-stub"]
    github_env = deployments[0].envs["github-mcp-stub"]
    assert isinstance(slack_env, DeployedAppEnvironment)
    assert isinstance(github_env, DeployedAppEnvironment)
    slack_endpoint = slack_env.deployed_app.endpoint
    github_endpoint = github_env.deployed_app.endpoint
    print(f"Slack MCP app: {slack_endpoint}")
    print(f"GitHub MCP app: {github_endpoint}")

    run = flyte.run(
        shepherd,
        prompt=args.prompt,
        slack_mcp_url=_mcp_session_url(slack_endpoint),
        github_mcp_url=_mcp_session_url(github_endpoint),
    )
    print(f"Run URL: {run.url}")
    run.wait()
    print(run.outputs())
