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

Once deployed, point your external system at the resulting ``/trigger`` URL:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"repository": "flyteorg/flyte", "pull_request": {"number": 123}, "action": "opened"}' \
    https://<subdomain>.apps.<endpoint>/trigger
```
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

tool_env = flyte.TaskEnvironment(
    name="webhook-agent-tools",
    image=(
        flyte.Image.from_debian_base(flyte_version="2.3.6")
        .with_apt_packages("git")
        .with_pip_packages("litellm", "httpx")
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@tool_env.task
async def fetch_pr(repo: str, number: int) -> dict[str, str]:
    """Fetch review-relevant metadata for a specific GitHub pull request."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        resp = await client.get(f"https://api.github.com/repos/{repo}/pulls/{number}")
        resp.raise_for_status()
        pr = resp.json()

    # Project the (large, deeply-nested, null-heavy) GitHub payload down to a
    # flat dict of strings. Returning the raw ``dict[str, Any]`` fails to
    # serialize because Flyte pickles each ``Any`` value and the payload is full
    # of ``None`` fields (assignee, milestone, merged_by, …), which cannot be
    # pickled.
    return {
        "title": str(pr.get("title") or ""),
        "state": str(pr.get("state") or ""),
        "author": str((pr.get("user") or {}).get("login") or ""),
        "base": str((pr.get("base") or {}).get("ref") or ""),
        "head": str((pr.get("head") or {}).get("ref") or ""),
        "additions": str(pr.get("additions") or 0),
        "deletions": str(pr.get("deletions") or 0),
        "changed_files": str(pr.get("changed_files") or 0),
        "url": str(pr.get("html_url") or ""),
        "body": str(pr.get("body") or ""),
    }


@tool_env.task
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


@tool_env.task(report=True)
async def review_pr(repo: str, pr_number: int, event: str) -> str:
    """Durable task that runs the agent for a single webhook event."""
    message = f"GitHub webhook fired for {repo}#{pr_number} (event={event}). Fetch the PR and post a review comment."
    result = await agent.run.aio(message)
    return result.summary or result.error


# ---------------------------------------------------------------------------
# FastAPI app that the webhook calls
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app):
    await flyte.init_passthrough.aio(
        project=flyte.current_project(),
        domain=flyte.current_domain(),
    )
    yield


def _build_app() -> Any:
    from fastapi import FastAPI

    from flyte.app.extras import FastAPIPassthroughAuthMiddleware

    api = FastAPI(title="flyte-agent-webhook", lifespan=_lifespan)

    # ``init_passthrough`` only forwards credentials that are bound to the Flyte
    # auth-metadata context for the current request. This middleware extracts the
    # incoming ``Authorization`` header and binds it, so ``flyte.run.aio`` can
    # authenticate its control-plane calls. Without it, every run submission
    # fails with "Failed to get signed url".
    api.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/health"})

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @api.post("/trigger")
    async def trigger(payload: dict) -> dict[str, str]:
        repo = payload.get("repository")
        pr_number = int(payload.get("pull_request", {}).get("number", 0))
        event = payload.get("action")

        run = await flyte.run.aio(review_pr, repo=repo, pr_number=pr_number, event=event)
        return {"run_url": run.url, "name": run.name}

    return api


webhook_env = flyte.app.AppEnvironment(
    name="flyte-agent-webhook",
    image=(flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "litellm")),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    depends_on=[tool_env],
)


@webhook_env.server
async def serve():
    import uvicorn

    config = uvicorn.Config(_build_app(), host="0.0.0.0", port=webhook_env.get_port().port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    import os

    import httpx

    import flyte.remote as remote
    from flyte.app import DeployedAppEnvironment

    flyte.init_from_config()
    deployments = flyte.deploy(webhook_env)
    print(f"Webhook agent deployed: {deployments[0].summary_repr()}")
    deployed_env = deployments[0].envs["flyte-agent-webhook"]
    assert isinstance(deployed_env, DeployedAppEnvironment)
    print(f"Webhook agent URL: {deployed_env.deployed_app.url}")

    app_handle = remote.App.get(name="flyte-agent-webhook")
    print(f"Webhook agent endpoint: {app_handle.endpoint}")

    api_key = os.environ.get("FLYTE_API_KEY")
    if not api_key:
        raise ValueError("FLYTE_API_KEY not set. Obtain with: flyte get api-key")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "flyte-webhook-client/1.0",
    }

    with app_handle.ephemeral_ctx_sync():
        with httpx.Client(headers=headers) as client:
            response = client.post(
                f"{app_handle.endpoint}/trigger",
                json={"repository": "flyteorg/flyte", "pull_request": {"number": 123}, "action": "opened"},
            )
            response.raise_for_status()
            print(f"Response: {response.json()}")
