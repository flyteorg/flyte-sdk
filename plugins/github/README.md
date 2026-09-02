# flyteplugins-github

Receive GitHub webhooks in Flyte.

```bash
pip install "flyteplugins-github[app]"
```

## Using it

Hand a `GitHubProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, run_once
from flyteplugins.github import GitHubProvider, events

# GitHubProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="github-webhooks", providers=[GitHubProvider()])


@app_env.on_event(events.PullRequest.OPENED)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    try:
        run = await run_once.aio(task, key=event.dedupe_key(), resource=event.resource_id)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}


flyte.serve(app_env)
```

Handlers must `await run_once.aio(...)`. The blocking form stalls the
app's event loop, and GitHub times deliveries out in seconds.

One app can serve several products at once — hand it one provider per product.

## Human review gates

`review_pr` parks a run on a `flyte.new_condition` carrying the pull request's
metadata as JSON, waits for a human to answer in the Flyte UI, and returns a
typed decision the workflow branches on:

```python
from flyteplugins.github import review_pr


@env.task
async def gated_merge(repo: str, number: int) -> str:
    decision = await review_pr(repo, number)
    if not decision.is_approved:
        return f"blocked: {decision.summary}"
    ...  # merge, with PyGithub
    return "merged"
```

The reviewer answers in markdown; `parse_review_payload` accepts raw JSON, a
fenced block, or JSON buried in prose, and normalizes verdict synonyms
(`lgtm`, `approved`, `changes_requested`, ...) — because people paste all of
those.

This lives in the plugin because the condition is the part only Flyte can do.
Reading the pull request is `PyGithub`'s job, which the gate calls directly
rather than wrapping:

```bash
pip install "flyteplugins-github[review]"
```

## Try it

`examples/github_webhooks.py` runs two ways. The first needs no GitHub account:

```bash
python examples/github_webhooks.py --local   # replay a real sample delivery in-process
python examples/github_webhooks.py           # deploy the receiver to Flyte
```

`--local` posts this plugin's `SAMPLE_DELIVERY` through the app with FastAPI's
test client, so you see a delivery verified, normalized, and dispatched — plus
an unsigned one refused with a 401, and the same delivery replayed to show the
dedupe key is stable.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret GITHUB_WEBHOOK_SECRET --value <secret>
   ```
2. Point GitHub at `<app-url>/webhook/github`, from
   repository Settings → Webhooks → Add webhook, content type `application/json`.

GitHub sends a `ping` when the webhook is created; it is answered automatically, so a green check in *Recent Deliveries* means the app is reachable.

**Verification:** HMAC-SHA256 over the raw body (`X-Hub-Signature-256`).

Comment and review events fold the comment id into `resource_id`, so two comments on one issue are two events rather than a redelivery of the first.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Call the GitHub API. Use `PyGithub` directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns only the part that is
Flyte's: authenticating an inbound delivery and turning it into a run.
