"""Webhook-triggered Agent — kick off the agent loop on external events.

Pattern
-------

This example wires up a small FastAPI app (deployed via ``flyte.app``) that
exposes a ``POST /trigger`` endpoint. When an external service (a GitHub
webhook, a paging tool, your CI, …) POSTs an event payload, the app launches
a fresh agent run.

The agent itself lives in a Flyte task so each invocation is durable and
observable in the Flyte console.

Deploy::

    flyte deploy examples/agents/flyte_agent/webhook_agent.py

Once deployed, point your external system at the resulting ``/trigger`` URL.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import flyte
import flyte.app
from flyte.ai.agents import Agent

# ---------------------------------------------------------------------------
# Agent + tools
# ---------------------------------------------------------------------------

task_env = flyte.TaskEnvironment(
    name="webhook-agent-tools",
    image=(flyte.Image.from_debian_base().with_pip_packages("litellm", "httpx")),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@task_env.task
async def fetch_pr(repo: str, number: int) -> dict[str, Any]:
    """Fetch metadata for a specific GitHub pull request."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        resp = await client.get(f"https://api.github.com/repos/{repo}/pulls/{number}")
        resp.raise_for_status()
        return resp.json()


@task_env.task
async def post_comment(repo: str, number: int, comment: str) -> str:
    """Post a comment on a GitHub issue or PR (stub)."""
    flyte.logger.info("Would post on %s#%d: %s", repo, number, comment)
    return "ok"


agent = Agent(
    name="pr-reviewer",
    instructions=(
        "You are a code-review assistant. Given a webhook event for a pull "
        "request, fetch the PR metadata, summarize the change in a single "
        "paragraph, and post a comment on the PR with your review."
    ),
    model="claude-haiku-4-5",
    tools=[fetch_pr, post_comment],
    max_turns=10,
)


@task_env.task(report=True)
async def review_pr(repo: str, pr_number: int, event: str) -> str:
    """Durable task that runs the agent for a single webhook event."""
    message = f"GitHub webhook fired for {repo}#{pr_number} (event={event}). Fetch the PR and post a review comment."
    result = await agent.run(message)
    return result.summary or result.error


# ---------------------------------------------------------------------------
# FastAPI app that the webhook calls
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]


@asynccontextmanager
async def _lifespan(_app):
    await flyte.init_passthrough.aio(
        project=flyte.current_project(),
        domain=flyte.current_domain(),
    )
    yield


def _build_app() -> Any:
    assert FastAPI is not None, "fastapi must be installed to build the webhook app."
    api = FastAPI(title="flyte-agent-webhook", lifespan=_lifespan)

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @api.post("/trigger")
    async def trigger(payload: dict) -> dict[str, str]:
        # Minimal GitHub-style payload extraction. Adapt for your provider.
        repo = payload.get("repository", {}).get("full_name", "flyteorg/flyte")
        pr_number = int(payload.get("pull_request", {}).get("number", 0))
        event = payload.get("action", "unknown")

        run = await flyte.run.aio(review_pr, repo=repo, pr_number=pr_number, event=event)
        return {"run_url": run.url, "name": run.name}

    return api


webhook_env = flyte.app.AppEnvironment(
    name="flyte-agent-webhook",
    image=(
        flyte.Image.from_debian_base(install_flyte=False).with_pip_packages("fastapi", "uvicorn", "litellm", "flyte")
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    depends_on=[task_env],
)


@webhook_env.server
async def serve():
    import uvicorn

    config = uvicorn.Config(_build_app(), host="0.0.0.0", port=webhook_env.get_port().port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(webhook_env)
    print(f"Webhook agent deployed: {deployments[0].summary_repr()}")
