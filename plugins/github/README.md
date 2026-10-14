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
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.github import GitHubProvider, events

# GitHubProvider.default_secret_env is mounted for you.
app_env = WebhookAppEnvironment(name="github-webhooks", providers=[GitHubProvider()])


@app_env.on_event(events.PullRequest.OPENED)
async def handle(event: WebhookEvent):
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

1. Invent a shared secret — nothing generates it for you — and store it under
   the name the provider mounts, `GITHUB_WEBHOOK_SECRET`:
   ```bash
   openssl rand -hex 32
   flyte create secret GITHUB_WEBHOOK_SECRET --value <secret>
   ```
2. Point GitHub at `<app-url>/webhook/github`, from
   repository Settings → Webhooks → Add webhook, pasting that same string into
   the webhook's **Secret** field. (A GitHub App has its own *Webhook* section
   with the same two fields; either route reaches the same endpoint.) A
   mismatch reads as a 401 in *Recent Deliveries*. Either content type works —
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

```bash
pip install "flyteplugins-github[auth]"
```

Configuration comes from three environment variables — `GITHUB_APP_ID`,
`GITHUB_APP_INSTALLATION_ID`, and `GITHUB_APP_PRIVATE_KEY`. What follows is
where those values come from, how to store them, how to get them onto a task,
and what happens when they are absent.

### Where the three values come from

Create the app first, from **Settings → Developer settings → GitHub Apps → New
GitHub App** ([github.com/settings/apps/new](https://github.com/settings/apps/new);
for an org, **Organization settings → Developer settings → GitHub Apps**). Give
it only the permissions the agent uses — *Contents: Read* to clone, *Read and
write* to push, *Pull requests: Read and write* to open PRs, *Issues: Read and
write* to comment. Then:

| Value | Where |
| --- | --- |
| `GITHUB_APP_ID` | The app's **General** tab, under *About* → **App ID**. A short number. |
| `GITHUB_APP_PRIVATE_KEY` | Same tab, **Private keys** → *Generate a private key*. Downloads a `.pem` file. |
| `GITHUB_APP_INSTALLATION_ID` | Install the app (**Install App** tab), then read the number off the URL you land on. |

Two things that reliably cost an hour:

- **The App ID is not the Client ID.** The General tab shows both, and the
  Client ID (`Iv23li...`) sits right next to the App ID. Signing a JWT with the
  Client ID as the issuer fails at GitHub, not locally, so the error surfaces as
  a failed mint rather than a bad value.
- **The private key is shown once.** GitHub hands over the `.pem` at generation
  and never again; losing it means generating a new one and deleting the old.
  Store it in Flyte before deleting the download.

The installation id is the fiddly one — it identifies *the app installed on one
account*, not the app. After installing, the browser lands on
`https://github.com/settings/installations/<id>` (or
`https://github.com/organizations/<org>/settings/installations/<id>`), and the
trailing number is it. To read it without the browser, ask the API — the app
JWT is the only credential this needs, so it works before any installation
token exists:

```python
import json, time, urllib.request
import jwt  # from flyteplugins-github[auth]

app_id, pem = "1234567", open("my-agent.private-key.pem").read()
now = int(time.time())
app_jwt = jwt.encode({"iat": now - 60, "exp": now + 600, "iss": app_id}, pem, algorithm="RS256")
request = urllib.request.Request(
    "https://api.github.com/app/installations",
    headers={"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github+json"},
)
for installation in json.load(urllib.request.urlopen(request)):
    print(installation["id"], installation["account"]["login"])
```

### Storing them as Flyte secrets

```bash
flyte create secret github-app-id --value 1234567
flyte create secret github-app-installation-id --value 87654321
flyte create secret github-app-private-key --from-file ~/Downloads/my-agent.private-key.pem
```

Use `--from-file` for the key, not `--value`: a PEM is multi-line, and a shell
that folds or strips those newlines produces a key that fails to parse at mint
time. `--from-file` stores the bytes verbatim, which is what the RS256 signer
needs. GitHub's download is a PKCS#1 PEM (`-----BEGIN RSA PRIVATE KEY-----`) and
is used as-is; no conversion step.

The app id and installation id are identifiers rather than credentials —
nothing breaks if they leak — but keeping all three in one place means one
mounting story instead of two.

`flyte create secret` writes to the org unless `--project`/`--domain` scope it.
Scope the secrets the same way as the environment that reads them, or leave all
of them org-wide; a secret in the wrong project is invisible at run time.

### Mounting them on the environment

Name the secrets in kebab-case and the env vars come out right on their own:
`flyte.Secret` upper-cases the key and turns `-` into `_`, so
`github-app-private-key` mounts as `GITHUB_APP_PRIVATE_KEY`, which is exactly
what `mint_installation_token()` reads.

```python
import flyte

env = flyte.TaskEnvironment(
    name="github-agent",
    secrets=[
        flyte.Secret("github-app-id"),
        flyte.Secret("github-app-installation-id"),
        flyte.Secret("github-app-private-key"),
    ],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github[auth]"),
)
```

Under any other naming, spell the target out — the env var is the contract, the
secret name is not:

```python
flyte.Secret("prod-bot-key", as_env_var="GITHUB_APP_PRIVATE_KEY")
```

Pass the same `secrets=[...]` to a `WebhookAppEnvironment` when a handler mints
tokens directly rather than launching a task that does.

### When they are missing

`GITHUB_TOKEN`/`GH_TOKEN` are honored as fallbacks, so a deployment can migrate
one secret at a time; once the app secrets exist the fallback never fires. A
deployment with none of them gets `None` back — with a logged reason — rather
than a crash, so unauthenticated paths keep working. That means a missing
secret shows up as *unauthenticated behavior*, not an exception: if clones of
private repos 404, check that all three landed.

## Event constants

`events` spells every event this plugin can dispatch, as `str` enums grouped by
event type, so a typo fails at import rather than by silently never matching.
GitHub grows actions faster than any constant list; for one the constants do
not cover yet, qualify the bare type with `action=`:

```python
@app_env.on_event(events.PullRequest.ANY, action="auto_merge_enabled")
```

(Raw qualified strings like `"pull_request.auto_merge_enabled"` work too.)

`event.payload` is GitHub's JSON verbatim, typed as `dict[str, Any]`. For
autocomplete, take a typed view of it — a cast, not a copy or a validation:

```python
payload = payloads.pull_request(event)  # payload["pull_request"]["head"]["ref"] completes
payload = payloads.issue_comment(event)  # payload["comment"]["body"], payload["issue"]["number"], ...
```

## What this plugin does not do

Wrap the GitHub API. Use `PyGithub` directly from your tasks — see
`examples/external_saas_integrations`. This plugin owns the parts every
GitHub agent otherwise duplicates: authenticating an inbound delivery and
turning it into a run, gating a run on a human review, and minting the App
token the outbound side authenticates with.
