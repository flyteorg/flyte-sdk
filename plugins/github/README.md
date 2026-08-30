# Flyte GitHub Plugin

Read and write GitHub from Flyte tasks, react to GitHub webhook events with an
app environment, gate workflows on human PR reviews, and expose everything as
an MCP server for agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-github"            # client only
pip install "flyteplugins-github[app]"       # + FastAPI app environment
pip install "flyteplugins-github[mcp]"       # + MCP server
```

## Setup

The plugin reads credentials from environment variables, which on Flyte are
populated by mounting secrets:

```bash
flyte create secret GITHUB_TOKEN --value <your-token>
flyte create secret GITHUB_WEBHOOK_SECRET --value <random-string>   # only for webhooks
```

A fine-grained personal access token scoped to your repositories with
*Pull requests: read & write*, *Issues: read & write*, *Contents: read &
write*, and *Checks: read & write* is enough for everything in this plugin; a
classic token needs the `repo` and `read:org` scopes.

Request the secret on any task or app environment that needs it:

```python
env = flyte.TaskEnvironment(
    name="github-demo",
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.github import GitHubClient

@env.task
async def summarize_pr(repo: str, number: int) -> str:
    async with GitHubClient() as client:
        pr = await client.get_pull_request(repo, number)
        files = await client.get_pull_request_files(repo, number)
    return f"{pr['title']}: {len(files)} files changed"
```

The client covers repositories, files, commits, issues, pull requests,
reviews, branches, check runs, and merging — see `flyteplugins.github.GitHubClient`.

## Human review gate (condition with a JSON payload)

`review_pr` parks a run on a `flyte.new_condition` whose markdown prompt
carries the PR's review metadata (files, diff stats, prior reviews) as an
embedded JSON block. The reviewer answers in the Flyte UI with JSON; the task
parses it back into a typed `ReviewDecision` it can branch on:

```python
from flyteplugins.github import review_pr, GitHubClient

@env.task
async def gated_merge(repo: str, number: int) -> str:
    decision = await review_pr(repo, number)
    if not decision.is_approved:
        return f"blocked: {decision.summary}"
    async with GitHubClient() as client:
        await client.merge_pull_request(repo, number, merge_method="squash")
    return "merged"
```

`parse_review_payload` is tolerant: it accepts raw JSON, fenced code blocks,
or JSON embedded in prose, and normalizes verdict synonyms (`lgtm` →
`approve`, `changes_requested` → `request_changes`, ...). Use
`collect_review_context` and `build_review_prompt` to build your own
condition flow.

## React to GitHub events

`GitHubAppEnvironment` serves a **setup dashboard** (`/`) and a **webhook
receiver** (`/webhook`). The dashboard walks through token creation, secret
creation, and repository webhook configuration; `/api/status` and
`/api/verify` expose machine-readable health.

```python
import flyte
from flyteplugins.github import GitHubAppEnvironment, launch_task

app_env = GitHubAppEnvironment(
    name="github-integration",
    secrets=[
        flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN"),
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event("pull_request.opened")
async def triage_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_pr", auto_version="latest")
    run = launch_task(task, key=event.dedupe_key(), repo=event.repository, number=event.number)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

Webhook payloads are HMAC-verified against `GITHUB_WEBHOOK_SECRET`
(`X-Hub-Signature-256`), normalized into `GitHubEvent` objects, matched
against the optional `repos` allowlist, and dispatched to handlers registered
with `on_event` (event types like `pull_request` or qualified
`pull_request.opened`; an empty pattern matches everything).

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event, and a second delivery of the same event raises
`DuplicateRun` instead of launching a second run. Failed or aborted runs never
block, so re-triggering after a failure is a retry.

Point a repository webhook (Settings → Webhooks) at the app's public URL +
`/webhook`, content type `application/json`, with the same secret value.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use GitHub through the Model Context Protocol:

```python
import flyte
from flyteplugins.github import github_mcp_app_env

mcp_env = github_mcp_app_env(
    "github-mcp",
    secrets=[flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN")],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default**. Pass `read_only=False` to include
write tools, and `include_destructive=True` to additionally expose
`merge_pull_request`. Tool annotations (`readOnlyHint`, `destructiveHint`,
`idempotentHint`) are set from the tool registry so MCP clients can reason
about safety. Reacting to events is intentionally *not* an MCP tool — that is
the app environment's job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="github-agent",
    mcp_servers=[MCPServerSpec(name="github", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.github.Config` controls token/webhook-secret env var names, the
API base URL (GitHub Enterprise Server), timeouts, and retries. The module
exports `default_config`; pass a custom `Config` to `GitHubClient`,
`build_mcp_server`, or `collect_review_context` when you need it.
