# SaaS integration plugins

Receive SaaS webhooks in Flyte and launch runs from them.

The receiver lives in `flyte.extras.webhooks`; each product is its own package
contributing one `Provider`:

| Package | Product | Verification |
| --- | --- | --- |
| [`flyteplugins-github`](../github) | GitHub | HMAC-SHA256 (`X-Hub-Signature-256`) |
| [`flyteplugins-slack`](../slack) | Slack Events API | HMAC-SHA256 with a replay window (`X-Slack-Signature`) |
| [`flyteplugins-linear`](../linear) | Linear | HMAC-SHA256 (`X-Linear-Signature`) |
| [`flyteplugins-clickup`](../clickup) | ClickUp | HMAC-SHA256 (`X-Clickup-Signature`) |
| [`flyteplugins-jira`](../jira) | Jira Cloud | none — Jira does not sign; a shared token stands in |

Install core plus the packages for the products you wire up — each row above is
a distribution name:

```bash
pip install "flyteplugins-github[app]"
```

## Try one without an account

Every plugin ships an example that replays a real sample delivery through the
app in-process — verification, normalization, and dispatch, with nothing to
configure:

```bash
python plugins/webhooks/github/examples/github_webhooks.py --local
```

## The division of labor

**`flyte.extras.webhooks` owns** the app, the dashboard, dispatch, the scope
allowlist, idempotent launching, the normalized event, and the verification
primitives — the parts that are easy to get subtly wrong and expensive to get
wrong once per product. It ships with flyte, and adds no runtime dependency:
serving the app needs `fastapi`, which stays an optional extra.

**A provider plugin owns** only what is specific to its product: which
environment variable holds the secret, how to verify a delivery, how to parse
one into a `WebhookEvent`, and the typed constants for its events. That is
usually under 150 lines.

**Calling the product's API** is where these packages have room to grow. Today
they own webhooks only, and the recipes in `examples/external_saas_integrations`
call `PyGithub`, `slack_sdk`, and the rest directly. A plugin earns a client
method when it does something the vendor SDK cannot — return a `flyte.io.File`
instead of an inline megabyte diff, render into the task report, or participate
in caching and fan-out. Forwarding arguments and reshaping JSON does not.

## One app, many products

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment
from flyteplugins.github import GitHubProvider
from flyteplugins.github import events as github_events
from flyteplugins.slack import SlackProvider

app_env = WebhookAppEnvironment(name="saas-webhooks", providers=[GitHubProvider(), SlackProvider()])


@app_env.on_event(github_events.PullRequest.OPENED)
async def triage(event): ...
```

Each provider gets a route at `/webhook/<name>`; anything not configured 404s.
The dashboard at `/` shows one row per provider with its payload URL, whether
its secret is mounted, and how it is verified.

## Conformance

Every provider plugin ships the same one-line test:

```python
from flyte.extras.webhooks.testing import assert_provider_conforms
import flyteplugins.github as plugin


def test_conformance():
    assert_provider_conforms(plugin)
```

CI fails if a plugin drifts. The harness checks the things that actually go
wrong: a verifier that raises instead of returning False on a hostile header,
event constants that render as `Class.MEMBER` rather than their wire value, a
dedupe key that is unstable, a sample delivery that no constant spells.

It leans on `SAMPLE_DELIVERY`, a real payload each plugin ships. Without one
there is no way to assert that `verify` and `parse` agree with the product
rather than merely with each other.

## Adding a product

Copy the smallest existing plugin, implement `verify` and `parse`, export a
`Provider` subclass with its defaults pre-wired, `events`, and
`SAMPLE_DELIVERY`, and add the conformance test. `examples/apps/webhook_custom_provider.py`
is a complete worked version you can run.
