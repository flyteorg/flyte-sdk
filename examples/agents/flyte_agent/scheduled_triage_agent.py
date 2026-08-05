"""Scheduled triage agent — wakes up daily, runs durable Flyte tasks as tools.

Pattern
-------

This example shows how to operate a :class:`flyte.ai.agents.Agent` as a
**scheduled, autonomous workflow**. The agent:

1. Runs every day at 9 AM (Cron trigger).
2. Calls durable ``@env.task`` tools to pull open issues, classify them by
   severity, and post a digest to the team channel.
3. Returns a short summary that is captured as the task output and shown in
   the Flyte console.

The "wakeup" is a regular Flyte task — the agent loop runs inside it, so every
tool call is durable, observable, and retryable on its own.

Deploy::

    flyte deploy examples/agents/flyte_agent/scheduled_triage_agent.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

import flyte
from flyte.ai.agents import Agent

# ---------------------------------------------------------------------------
# Tools as durable Flyte tasks
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="triage-agent",
    image=(flyte.Image.from_debian_base().with_pip_packages("httpx", "litellm")),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def list_open_issues(repo: str, max_count: int = 25) -> list[dict[str, str]]:
    """Fetch open issues for a GitHub repository.

    Uses the Search API so pull requests are excluded server-side; otherwise
    ``/repos/{repo}/issues`` returns issues *and* PRs mixed together, and busy
    repos can fill the first page with PRs.

    Args:
        repo: ``"owner/repo"`` slug, e.g. ``"flyteorg/flyte"``.
        max_count: Maximum number of issues to return (capped at 100 by the API).
    """
    import httpx

    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        resp = await client.get(
            "https://api.github.com/search/issues",
            params={
                "q": f"repo:{repo} is:issue is:open",
                "sort": "created",
                "order": "desc",
                "per_page": min(max_count, 100),
            },
        )
        resp.raise_for_status()
        issues = resp.json().get("items", [])
    return [
        {
            "number": str(i["number"]),
            "title": i["title"],
            "url": i["html_url"],
            "labels": ", ".join(label["name"] for label in i.get("labels", [])),
            "comments": str(i.get("comments", 0)),
        }
        for i in issues
    ]


@env.task
async def classify_issue(
    title: str,
    body: str = "",
) -> Literal["urgent", "important", "normal", "noise"]:
    """Score one issue by severity using simple heuristics.

    A real implementation would call your classifier model or use embeddings;
    we keep this stub deterministic so the example is self-contained.
    """
    text = f"{title} {body}".lower()
    if any(kw in text for kw in ("critical", "outage", "data loss", "security", "p0")):
        return "urgent"
    if any(kw in text for kw in ("bug", "regression", "broken", "p1", "p2")):
        return "important"
    if any(kw in text for kw in ("typo", "nit", "tracking")):
        return "noise"
    return "normal"


@env.task
async def post_digest(channel: str, summary: str) -> dict[str, str]:
    """Post a Markdown digest to a chat channel (stub — replace with Slack/Linear/etc.)."""
    flyte.logger.info("Posting digest to %s:\n%s", channel, summary)
    return {"channel": channel, "delivered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


# ---------------------------------------------------------------------------
# Agent + scheduled entrypoint
# ---------------------------------------------------------------------------

agent = Agent(
    name="github-triage",
    instructions=(
        "You are a GitHub issue triager. For each wakeup:\n"
        "- list open issues for the configured repo,\n"
        "- classify each one with classify_issue,\n"
        "- group them by severity, and\n"
        "- post a concise digest to the team channel.\n"
        "Always end by calling post_digest."
    ),
    model="claude-haiku-4-5",
    tools=[list_open_issues, classify_issue, post_digest],
    max_turns=20,
)


@env.task(
    triggers=flyte.Trigger(
        "daily-triage",
        flyte.FixedRate(1),  # run every 1 minutes
        inputs={"trigger_time": flyte.TriggerTime, "repo": "flyteorg/flyte", "channel": "#flyte-triage"},
    ),
    report=True,
)
async def triage_repo(trigger_time: datetime, repo: str, channel: str) -> str:
    """Daily scheduled wakeup that runs the triage agent end-to-end."""
    message = f"It is {trigger_time.isoformat()}. Triage the open issues in {repo} and post a digest to {channel}."
    with flyte.group("triage-loop"):
        result = await agent.run.aio(message)
    if result.error:
        return f"[triage failed] {result.error}"
    return result.summary


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
