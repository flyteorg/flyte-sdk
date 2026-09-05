# flyteplugins-github

Receive GitHub webhooks in Flyte — JSON or form-encoded, GitHub signs both the
same way — plus human review gates on pull requests and GitHub App
installation tokens for agents that clone, push, or open PRs.

```bash
pip install "flyteplugins-github[app]"
```

## Using it

Hand a `GitHubProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.github import GitHubProvider, events

# GitHubProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="github-webhooks", providers=[GitHubProvider()])


@app_env.on_event(events.PullRequest.OPENED)
async def handle(event):
    import flyte.remote as remote

    task = remote.Task.get(name="my-env.my_task", auto_version="latest")
    result = await run_once.aio(task, key=event.dedupe_key(), resource=event.resource_id)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


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
an unsigned one refused with a 401, the same delivery replayed to show the
dedupe key is stable, and the same delivery form-encoded (GitHub's default
content type) landing on that same key.

## Setup

1. Store the secret and mount it on the app:
   ```bash
   flyte create secret GITHUB_WEBHOOK_SECRET --value <secret>
   ```
2. Point GitHub at `<app-url>/webhook/github`, from
   repository Settings → Webhooks → Add webhook. Either content type works —
   the form's default `application/x-www-form-urlencoded` wraps the JSON in a
   `payload=` field and is unwrapped automatically; `application/json` keeps
   the deliveries readable in *Recent Deliveries*.

GitHub sends a `ping` when the webhook is created; it is answered automatically, so a green check in *Recent Deliveries* means the app is reachable.

**Verification:** HMAC-SHA256 over the raw body (`X-Hub-Signature-256`), whichever content type the webhook uses.

Comment and review events fold the comment id into `resource_id`, so two comments on one issue are two events rather than a redelivery of the first.

## GitHub App tokens

Agents that clone, push, or open PRs authenticate best as a GitHub App: hold
no personal access token, mint a short-lived installation token per operation.
Tokens live one hour — plenty for a clone or a `gh pr create`, useless to an
attacker who exfiltrates one from a log:

```python
import asyncio

from flyteplugins.github import clone_url, mint_installation_token


@env.task
async def open_fix_pr(repo: str) -> str:
    # One HTTPS round trip; keep it off the event loop.
    token = await asyncio.to_thread(mint_installation_token)
    url = clone_url(repo, token)  # https://x-access-token:<token>@github.com/...
    ...
```

Configuration comes from three secrets, mounted as environment variables on
the task's environment — `GITHUB_APP_ID`, `GITHUB_APP_INSTALLATION_ID`, and
`GITHUB_APP_PRIVATE_KEY` (the app's PEM key). `GITHUB_TOKEN`/`GH_TOKEN` are
honored as fallbacks so a deployment can migrate one secret at a time, and a
deployment with none of them gets `None` back — with a logged reason — rather
than a crash, so unauthenticated paths keep working.

```bash
pip install "flyteplugins-github[auth]"
```

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
Raw strings still work, for events the constants do not cover yet.

## What this plugin does not do

Wrap the GitHub API. Use `PyGithub` directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns the parts every
GitHub agent otherwise duplicates: authenticating an inbound delivery and
turning it into a run, gating a run on a human review, and minting the App
token the outbound side authenticates with.
