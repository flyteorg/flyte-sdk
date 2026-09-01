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
from flyteplugins.github import GitHubClient, events

@env.task
async def summarize_pr(repo: str, number: int) -> str:
    async with GitHubClient() as client:
        pr = await client.get_pull_request.aio(repo, number)
        files = await client.get_pull_request_files.aio(repo, number)
    return f"{pr['title']}: {len(files)} files changed"
```

The client covers repositories, files, commits, issues, pull requests,
reviews, branches, check runs, and merging — see `flyteplugins.github.GitHubClient`.

### Both call forms

Every client method is available two ways. `await client.get_pull_request.aio(...)` is the
async form — use it in `async def` tasks and anywhere on an app's event loop.
`client.get_pull_request(...)` is the blocking form, for plain `def` tasks and scripts:

```python
@env.task
def summarize(...) -> str:
    with GitHubClient() as client:          # note: `with`, not `async with`
        pr = client.get_pull_request(repo, number)
    ...
```

The blocking form parks the calling thread until the call returns, so never
reach for it inside an `async def` task or a webhook handler — it would stall
the event loop and everything else waiting on it.

## Human review gate (condition with a JSON payload)

`review_pr` parks a run on a `flyte.new_condition` whose markdown prompt
carries the PR's review metadata (files, diff stats, prior reviews) as an
embedded JSON block. The reviewer answers in the Flyte UI with JSON; the task
parses it back into a typed `ReviewDecision` it can branch on:

```python
from flyteplugins.github import GitHubClient, events, review_pr

@env.task
async def gated_merge(repo: str, number: int) -> str:
    decision = await review_pr(repo, number)
    if not decision.is_approved:
        return f"blocked: {decision.summary}"
    async with GitHubClient() as client:
        await client.merge_pull_request.aio(repo, number, merge_method="squash")
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
from flyteplugins.github import GitHubAppEnvironment, events, launch_task

app_env = GitHubAppEnvironment(
    name="github-integration",
    secrets=[
        flyte.Secret("GITHUB_TOKEN", as_env_var="GITHUB_TOKEN"),
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
    ],
)

@app_env.on_event(events.PullRequest.OPENED)
async def triage_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_pr", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), repo=event.repository, number=event.number)
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
`DuplicateRun` instead of launching a second run. Identity lives entirely on
that label — run names are left to the control plane. Failed or aborted runs
never block, so re-triggering after a failure is a retry. The key is just a
string: pass your own to choose a different idempotency scope.

Always `await launch_task.aio(...)` inside a handler. The synchronous
`launch_task(...)` form is for scripts: it blocks the calling thread, which on
the app's event loop stalls every other in-flight request.


Point a repository webhook (Settings → Webhooks) at the app's public URL +
`/webhook`, content type `application/json`, with the same secret value.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use GitHub through the Model Context Protocol:

```python
import flyte
from flyteplugins.github import events, github_mcp_app_env

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

## Testing

An end-to-end pass against a real repository. Use a scratch repo you own —
step 4 comments on and labels a real pull request.

**1. Create the credentials.**

A fine-grained personal access token (GitHub → Settings → Developer settings →
Fine-grained tokens) scoped to your test repo, with *Pull requests: read &
write*, *Issues: read & write*, and *Checks: read & write*. Then pick any
random string as the webhook secret:

```bash
flyte create secret GITHUB_TOKEN --value <token>
flyte create secret GITHUB_WEBHOOK_SECRET --value <random-string>
```

**2. Check the client works before involving the platform.** Everything below
is easier to debug once you know the token is good:

```bash
export GITHUB_TOKEN=<token>
python -c "
from flyteplugins.github import GitHubClient
with GitHubClient() as c:
    print(c.get_repository('<owner>/<repo>')['full_name'])
"
```

**3. Deploy the task the webhook will launch.**

```bash
flyte deploy plugins/github/examples/read_write_pr.py env
```

`react_to_pr_events.py` looks this task up by name (`triage_pr`), so it has to
exist before the app can launch it.

**4. Run the read/write task directly**, to confirm writes land before any
webhook is in play. Open a PR in your test repo, then:

```bash
flyte run plugins/github/examples/read_write_pr.py triage_pr \
    --repo <owner>/<repo> --number <pr-number>
```

The PR should pick up a `flyte-triage` label, a comment with its diff stats,
and a check run.

**5. Deploy the webhook app.**

```bash
python plugins/github/examples/react_to_pr_events.py
```

It prints the app URL. Open it: the dashboard should show both secrets as
mounted, and *Verify GitHub credentials* should return your login.

**6. Point GitHub at the app.** In the repo's Settings → Webhooks → Add
webhook:

- Payload URL: `<app-url>/webhook`
- Content type: `application/json`
- Secret: the same value you used for `GITHUB_WEBHOOK_SECRET`
- Events: *Let me select individual events* → Pull requests, Issues

GitHub immediately sends a `ping`, which the receiver answers with
`{"ok": true, "ping": true}` — a green checkmark in the webhook's *Recent
Deliveries* tab means the URL is reachable.

**7. Trigger a real event.** Open a new pull request. Then check, in order:

- GitHub's *Recent Deliveries* tab — the `pull_request` delivery should be 200,
  and the response body names the handler that ran and the run it launched.
- `<app-url>/api/events` — the normalized event.
- `flyte get runs` — a run whose `dedupe` label matches the event.
- The PR itself — label, comment, and check run.

**8. Confirm idempotency.** Hit *Redeliver* on that same delivery in GitHub.
The response should report `skipped` with a `DuplicateRun` message, and no
second run should appear. This is the behaviour worth checking by hand, since
it is the one a webhook sender will exercise on its own during an outage.

**9. Optional — the MCP server.**

```bash
python plugins/github/examples/github_mcp_server.py
claude mcp add --transport http github-mcp <app-url>/mcp/mcp
```

Ask an agent to summarize a PR in your test repo. The default surface is
read-only, so it can look but not touch.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Webhook delivery returns 401 | The secret in GitHub does not match `GITHUB_WEBHOOK_SECRET`. |
| Delivery returns 503 | `GITHUB_WEBHOOK_SECRET` is not mounted on the app; check `/api/status`. |
| Delivery is 200 but no run appears | No handler matched. `/api/status` lists the registered patterns. |
| Handler reports a task-not-found error | Step 3 was skipped, or the task deployed under a different name. |
| Second delivery launches a second run | Expected when the first run failed — failed runs do not block a retry. |
