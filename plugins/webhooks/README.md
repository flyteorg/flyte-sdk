# flyteplugins-webhooks

Receive SaaS webhooks in Flyte and launch runs from them.

One app environment accepts webhooks from **GitHub, Slack, Linear, ClickUp, and
Jira**, verifies each with that provider's own scheme, normalizes the payload
into a single `WebhookEvent`, and dispatches it to handlers you register.

```bash
pip install "flyteplugins-webhooks[app]"
```

## Receiving events

```python
import flyte
from flyte.extras import DuplicateRun, idempotent_run
from flyteplugins.webhooks import WebhookAppEnvironment, events

app_env = WebhookAppEnvironment(
    name="saas-webhooks",
    providers=["github", "slack"],
    secrets=[
        flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
        flyte.Secret("SLACK_SIGNING_SECRET", as_env_var="SLACK_SIGNING_SECRET"),
    ],
)


@app_env.on_event(events.github.PullRequest.OPENED)
async def triage_new_pr(event):
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    try:
        run = await idempotent_run.aio(task, key=event.dedupe_key(), repo=event.scope)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


flyte.serve(app_env)
```

Handlers must `await idempotent_run.aio(...)`. The blocking form stalls the
app's event loop, and webhook senders time deliveries out in seconds.

## What it serves

| Path | Purpose |
| --- | --- |
| `/` | Setup dashboard: per-provider payload URLs, whether each secret is mounted, recent events |
| `/webhook/{provider}` | Verified receiver. Providers you did not configure 404 |
| `/api/status` | Machine-readable status |
| `/api/events` | Recent events, newest first |
| `/healthz` | Liveness |

GitHub's `ping` and Slack's `url_verification` handshake are answered
automatically, so both go green on setup without a handler.

## The normalized event

Five payload shapes, one model. Handlers match on `qualified_type` and read the
fields they need; `payload` always carries the provider's original JSON.

| Field | Meaning |
| --- | --- |
| `provider` | `github`, `slack`, `linear`, `clickup`, `jira` |
| `event_type` / `action` | The provider's type and action; `qualified_type` joins them |
| `resource_id` | Issue key, task id, message timestamp — what the event is about |
| `occurred_at` | The provider's timestamp, when it sends one |
| `scope` | Repository, channel, team, list, or project key |
| `title`, `url`, `actor` | For dashboards and messages |
| `payload` | The original JSON, verbatim |

`event.dedupe_key()` combines provider, type, resource, and timestamp. The
timestamp is what makes it usable for `update`-shaped events: without it, every
later change to one resource would collapse onto the first one's key and never
launch. The key is just a string — build your own and pass it to
`idempotent_run` when you want a different scope, such as one run per Slack
thread rather than per message.

## Typed event constants

```python
from flyteplugins.webhooks import events

@app_env.on_event(events.github.PullRequest.ANY)      # every pull_request action
@app_env.on_event(events.github.PullRequest.OPENED)   # just pull_request.opened
@app_env.on_event(events.jira.Issue.CREATED)          # jira:issue_created
```

`str` enums grouped by event type, so a typo fails at import rather than by
silently never matching. Raw strings still work, for events the constants do not
cover yet.

## Verification

| Provider | Scheme |
| --- | --- |
| GitHub | HMAC-SHA256 over the raw body (`X-Hub-Signature-256`) |
| Slack | HMAC-SHA256 over `v0:{timestamp}:{body}`, with a five-minute replay window |
| Linear | HMAC-SHA256 over the raw body (`X-Linear-Signature`) |
| ClickUp | HMAC-SHA256 over the raw body (`X-Clickup-Signature`) |
| Jira | **Not signed.** Jira Cloud does not sign webhooks, so the receiver falls back to a shared token in `X-Webhook-Token` — which something in front of the app has to inject, since Jira cannot send custom headers |

Every comparison runs on bytes: `hmac.compare_digest` raises `TypeError` on
`str` operands containing non-ASCII, and these headers are attacker-controlled,
so a crafted one would otherwise turn a clean 401 into a 500.

## What this plugin does not do

Call the products' APIs. Use their own maintained clients — `PyGithub`,
`slack_sdk`, `jira`, `gql` — directly from your tasks. See
`examples/external_saas_integrations` for worked recipes of each. This plugin
owns only the part that is genuinely Flyte's: authenticating an inbound delivery
and turning it into a run.

## Testing

An end-to-end pass against real accounts. Use scratch repos, channels, and
projects — these steps write to them.

**1. Store a secret per provider you are enabling.**

```bash
flyte create secret GITHUB_WEBHOOK_SECRET  --value <random-string>
flyte create secret SLACK_SIGNING_SECRET   --value <Basic Information -> Signing Secret>
flyte create secret LINEAR_WEBHOOK_SECRET  --value <shown when you create the webhook>
flyte create secret CLICKUP_WEBHOOK_SECRET --value <shown when you create the webhook>
flyte create secret JIRA_WEBHOOK_TOKEN     --value <random-string you invent>
```

Linear and ClickUp only show their signing secret *after* you create the webhook
in step 4, so expect to come back and set those.

**2. Deploy the tasks the handlers launch.** They are looked up by name, so they
must exist before the app can launch them:

```bash
flyte deploy examples/external_saas_integrations/github_triage_pr.py env
flyte deploy examples/external_saas_integrations/slack_notify.py env
```

**3. Deploy the receiver.**

```bash
python examples/external_saas_integrations/webhook_receiver.py
```

It prints the app URL. Open it — the dashboard lists one payload URL per
provider and shows which secrets are mounted. Fix any missing ones before
going further; an unmounted secret means that provider's deliveries get a 503.

**4. Point each provider at its URL** from the dashboard table:

- **GitHub** — repo Settings → Webhooks → Add webhook. Content type
  `application/json`, same secret value, events *Pull requests* and *Issues*.
  GitHub sends a `ping` immediately; a green check in *Recent Deliveries* means
  the URL is reachable.
- **Slack** — api.slack.com/apps → Event Subscriptions. The Request URL field
  verifies itself via the `url_verification` handshake. Subscribe to
  `app_mention`, then invite the bot to a channel.
- **Linear** — Settings → API → Webhooks. Store the secret it shows you.
- **ClickUp** — Space Settings → Integrations → Webhooks. Store the secret.
- **Jira** — Settings → System → Webhooks. Jira cannot send the
  `X-Webhook-Token` header itself, so either put a gateway in front that injects
  it, or deploy with `require_signature=False` and protect the app at the
  network level. To confirm the path without a gateway:

  ```bash
  curl -X POST <app-url>/webhook/jira \
    -H 'Content-Type: application/json' -H 'X-Webhook-Token: <your token>' \
    -d '{"webhookEvent":"jira:issue_created","issue":{"key":"PROJ-1",
         "fields":{"summary":"hand-made","project":{"key":"PROJ"}}}}'
  ```

**5. Trigger a real event.** Open a pull request, mention the bot, create an
issue. Then check, in order:

- the provider's own delivery log — 200, with a body naming the handler that ran
- `<app-url>/api/events` — the normalized event
- `flyte get runs` — a run whose `dedupe` label matches
- the resource itself — the label, comment, or reply the task wrote

**6. Confirm idempotency.** Redeliver the same event from the provider's UI. The
response should report `skipped` with a `DuplicateRun` message and launch
nothing. Then make a *new* change to the same resource: because the dedupe key
folds in the provider's timestamp, that one does launch. This pair is the
behaviour worth seeing by hand — it is what a webhook sender will exercise on
its own during an outage.

**7. Optional — the scope allowlist.** Redeploy with
`scopes=["<your-repo-or-channel>"]` and trigger an event somewhere else. The
receiver answers 200 with a `skipped` message. Note it fails closed: an event
carrying no scope is skipped too, since an allowlist cannot vouch for something
it cannot attribute.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| 404 | That provider is not in the app's `providers` list |
| 401 | The secret does not match what the provider is signing with |
| 401 from Slack only sometimes | Clock skew — signatures older than five minutes are rejected as replays |
| 503 | That provider's secret is not mounted; `/api/status` says which |
| 200 but no run | No handler matched, or the allowlist skipped it — the response body says which |
| Handler reports task-not-found | Step 2 was skipped, or the task deployed under a different name |
| A redelivery launches a second run | Expected when the first run failed — failed runs do not block a retry |
