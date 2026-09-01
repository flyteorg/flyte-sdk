# Flyte Jira Plugin

Read and write Jira Cloud issues from Flyte tasks, react to Jira webhook
events with an app environment, and expose everything as an MCP server for
agents running on Flyte.

## Installation

```bash
pip install "flyteplugins-jira"            # client only
pip install "flyteplugins-jira[app]"       # + FastAPI app environment
pip install "flyteplugins-jira[mcp]"       # + MCP server
```

## Setup

Jira Cloud authenticates with your account email plus an API token created at
[id.atlassian.net/manage/profile/api-tokens](https://id.atlassian.net/manage/profile/api-tokens).
Store all three credentials as Flyte secrets:

```bash
flyte create secret JIRA_BASE_URL --value https://<site>.atlassian.net
flyte create secret JIRA_EMAIL --value you@example.com
flyte create secret JIRA_API_TOKEN --value <api-token>
flyte create secret JIRA_WEBHOOK_TOKEN --value <random-string>   # only for webhooks
```

Request the secrets on any task or app environment that needs them:

```python
env = flyte.TaskEnvironment(
    name="jira-demo",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)
```

## Read/write from tasks

```python
import flyte
from flyteplugins.jira import JiraClient, events

@env.task
async def open_ticket(project_key: str, summary: str) -> str:
    async with JiraClient() as client:
        issue = await client.create_issue.aio(project_key, summary)
    return issue["url"]
```

The client covers projects, issues, JQL search, comments, and workflow
transitions — see `flyteplugins.jira.JiraClient`. Plain-text descriptions and
comments are converted to Jira's Atlassian Document Format automatically, and
issue descriptions are converted back to plain text on read. Errors are raised
as `JiraAPIError`; 429 rate limits are retried.

### Both call forms

Every client method is available two ways. `await client.get_issue.aio(...)` is the
async form — use it in `async def` tasks and anywhere on an app's event loop.
`client.get_issue(...)` is the blocking form, for plain `def` tasks and scripts:

```python
@env.task
def summarize(...) -> str:
    with JiraClient() as client:          # note: `with`, not `async with`
        issue = client.get_issue(issue_key)
    ...
```

The blocking form parks the calling thread until the call returns, so never
reach for it inside an `async def` task or a webhook handler — it would stall
the event loop and everything else waiting on it.

## React to Jira events

`JiraAppEnvironment` serves a **setup dashboard** (`/`) and a **webhook
receiver** (`/webhook`). The dashboard walks through API token creation,
secret creation, and Jira webhook configuration; `/api/status` and
`/api/verify` expose machine-readable health.

```python
import flyte
from flyteplugins.jira import JiraAppEnvironment, events, launch_task

app_env = JiraAppEnvironment(
    name="jira-integration",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
        flyte.Secret("JIRA_WEBHOOK_TOKEN", as_env_var="JIRA_WEBHOOK_TOKEN"),
    ],
)

@app_env.on_event(events.Issue.CREATED)
async def triage_new_issue(event):
    import flyte.remote as remote

    task = remote.Task.get(name="triage_issue", auto_version="latest")
    run = await launch_task.aio(task, key=event.dedupe_key(), issue_key=event.issue_key)
    return {"run": run.name}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(app_env)
```

**Jira webhooks are not signed.** The receiver protects itself with a shared
token: choose a random string, store it as `JIRA_WEBHOOK_TOKEN`, and deliver
webhooks through a gateway or proxy that adds the `X-Webhook-Token` header
(Jira itself cannot attach custom headers). If that is not possible, set
`require_webhook_token=False` and protect the endpoint at the network level —
the dashboard explains both options.

Events are normalized into `JiraEvent` objects, matched against the optional
`project_keys` allowlist, and dispatched to handlers registered with
`on_event` (names like `jira:issue_created`, `jira:issue_updated`,
`comment_created`; an empty pattern matches everything).

`launch_task` launches runs **idempotently**: every run carries a `dedupe`
label derived from the event, and a second delivery of the same event raises
`DuplicateRun` instead of launching a second run. Identity lives entirely on
that label — run names are left to the control plane. Failed or aborted runs
never block, so re-triggering after a failure is a retry. The key is just a
string: pass your own to choose a different idempotency scope.

Always `await launch_task.aio(...)` inside a handler. The synchronous
`launch_task(...)` form is for scripts: it blocks the calling thread, which on
the app's event loop stalls every other in-flight request.


Create the webhook in Jira (gear → Products → Webhooks, site admins) pointing
at the app's public URL + `/webhook`.

## MCP server for agents

The read/write surface doubles as MCP tools, so agents running on Flyte can
use Jira through the Model Context Protocol:

```python
import flyte
from flyteplugins.jira import events, jira_mcp_app_env

mcp_env = jira_mcp_app_env(
    "jira-mcp",
    secrets=[
        flyte.Secret("JIRA_BASE_URL", as_env_var="JIRA_BASE_URL"),
        flyte.Secret("JIRA_EMAIL", as_env_var="JIRA_EMAIL"),
        flyte.Secret("JIRA_API_TOKEN", as_env_var="JIRA_API_TOKEN"),
    ],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.serve(mcp_env)
```

The server is **read-only by default** (projects, issues, search, comments,
transitions). Pass `read_only=False` to include issue creation, updates,
commenting, and transitions, and `include_destructive=True` to additionally
expose `delete_issue`. Tool annotations (`readOnlyHint`, `destructiveHint`,
`idempotentHint`) are set from the tool registry. Reacting to events is
intentionally *not* an MCP tool — that is the app environment's job.

Connect an agent running on Flyte:

```python
from flyte.ai.agents import Agent, MCPServerSpec

agent = Agent(
    name="jira-agent",
    mcp_servers=[MCPServerSpec(name="jira", url="https://<app>/mcp/mcp")],
)
```

## Configuration

`flyteplugins.jira.Config` controls the credential env var names, the REST API
path, timeouts, and retries. The module exports `default_config`; pass a custom
`Config` to `JiraClient`, `build_mcp_server`, or the app environment when you
need it.

## Testing

An end-to-end pass against a real Jira Cloud site. Use a scratch project —
step 4 creates and comments on real issues.

**1. Create the credentials.** An API token from
<https://id.atlassian.com/manage-profile/security/api-tokens>. Jira uses basic
auth over your account email plus the token, so all three values are secrets:

```bash
flyte create secret JIRA_BASE_URL  --value https://<your-site>.atlassian.net
flyte create secret JIRA_EMAIL     --value you@example.com
flyte create secret JIRA_API_TOKEN --value <api-token>
flyte create secret JIRA_WEBHOOK_TOKEN --value <random-string>
```

`JIRA_WEBHOOK_TOKEN` is a value you invent. Jira webhooks are **not signed**,
so the receiver authenticates them with this shared token instead — see step 6.

**2. Check the client works before involving the platform:**

```bash
export JIRA_BASE_URL=https://<your-site>.atlassian.net
export JIRA_EMAIL=you@example.com
export JIRA_API_TOKEN=<api-token>
python -c "
from flyteplugins.jira import JiraClient
with JiraClient() as c:
    print(c.get_myself()['displayName'])
    print([p['key'] for p in c.list_projects()])
"
```

A 401 here is almost always the email/token pair rather than the token alone.

**3. Deploy the task the webhook will launch.**

```bash
flyte deploy plugins/jira/examples/manage_ticket.py env
```

`react_to_jira_events.py` looks this task up by name (`triage_issue`), so it
has to exist before the app can launch it.

**4. Run a task directly**, to confirm writes land before any webhook is
involved:

```bash
flyte run plugins/jira/examples/manage_ticket.py open_ticket \
    --project_key <PROJ> --summary "Flyte test issue" --description "created by the plugin test"
```

**5. Deploy the webhook app.**

```bash
python plugins/jira/examples/react_to_jira_events.py
```

It prints the app URL. Open it: the dashboard should show all four secrets
mounted, and *Verify Jira credentials* should return your display name.

**6. Point Jira at the app.** Jira Settings → System → Webhooks → *Create a
Webhook*:

- URL: `<app-url>/webhook`
- Events: *Issue created* and *Issue updated*

Jira cannot add custom headers, and it does not sign its webhooks — so the
`X-Webhook-Token` header the receiver requires has to be injected by whatever
sits in front of the app (an API gateway, a reverse proxy, or a small
forwarder). For a first local test, deploy the app with
`require_webhook_token=False` and protect it at the network level instead,
then send a delivery by hand to confirm the path works:

```bash
curl -X POST <app-url>/webhook \
  -H 'Content-Type: application/json' \
  -H 'X-Webhook-Token: <the value you chose>' \
  -d '{"webhookEvent":"jira:issue_created","issue":{"key":"PROJ-1","fields":{"summary":"hand-made","project":{"key":"PROJ"}}}}'
```

**7. Trigger a real event.** Create an issue in the test project. Then check,
in order:

- `<app-url>/api/events` — the normalized event, `qualified_type` of
  `jira:issue_created`.
- `flyte get runs` — a run whose `dedupe` label matches.
- The issue — the triage task's comment.

**8. Confirm idempotency.** The dedupe key folds in Jira's event timestamp, so
a redelivered event dedupes while a later update to the same issue gets its own
run.

**9. Optional — the allowlist.** Redeploy with `project_keys=["<PROJ>"]` and
create an issue in another project. The receiver should answer 200 with a
`skipped` message. The allowlist fails closed, so an event carrying no project
key is skipped too.

**10. Optional — the MCP server.**

```bash
python plugins/jira/examples/jira_mcp_server.py
claude mcp add --transport http jira-mcp <app-url>/mcp/mcp
```

Ask an agent to summarize open bugs in the project. The default surface is
read-only.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Delivery returns 401 | The `X-Webhook-Token` header is missing or does not match `JIRA_WEBHOOK_TOKEN`. Jira alone cannot send it — see step 6. |
| Delivery returns 503 | `JIRA_WEBHOOK_TOKEN` is not mounted; check `/api/status`. |
| Client raises 401 | Basic auth needs the *email + token* pair, not the token alone. |
| A description shows as `[object Object]` | Jira v3 stores rich text as ADF; the client converts plain strings for you, so pass a plain string. |
| 200 but no run | No handler matched, or the allowlist skipped it — the response body says which. |
