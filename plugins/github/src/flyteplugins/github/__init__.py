"""GitHub webhooks for Flyte.

Hand a `GitHubProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import DuplicateRun, WebhookAppEnvironment, run_once
from flyteplugins.github import GitHubProvider, events

app_env = WebhookAppEnvironment(
    name="github-webhooks",
    providers=[GitHubProvider()],
    secrets=[flyte.Secret("GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET")],
)


@app_env.on_event(events.PullRequest.OPENED)
async def triage(event):
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    try:
        run = await run_once.aio(task, key=event.dedupe_key(), repo=event.scope)
    except DuplicateRun as exc:
        return {"skipped": str(exc)}
    return {"run": run.name}
```

## Human review gates

`review_pr` parks a run on a `flyte.new_condition` carrying the pull request's
metadata as JSON, waits for a human to answer in the Flyte UI, and returns a
typed decision:

```python
from flyteplugins.github import review_pr


@env.task
async def gated_merge(repo: str, number: int) -> str:
    decision = await review_pr(repo, number)
    if decision.is_approved:
        ...  # merge, with PyGithub
    return f"blocked: {decision.summary}"
```

It lives here because the condition is the part only Flyte can do. Reading the
pull request is `PyGithub`'s job, and this calls it directly rather than
wrapping it — install `flyteplugins-github[review]` for that extra.

Calling the GitHub API for anything else is not this plugin's job either; use
`PyGithub` from your tasks. See `examples/external_saas_integrations`.
"""

import hashlib
import hmac

from . import events
from ._provider import GitHubProvider, handshake, parse, verify
from ._review import (
    DEFAULT_TOKEN_ENV_VAR,
    ReviewComment,
    ReviewContext,
    ReviewDecision,
    Verdict,
    build_review_prompt,
    collect_review_context,
    condition_name_for,
    parse_review_payload,
    review_pr,
)

__all__ = [
    "DEFAULT_TOKEN_ENV_VAR",
    "SAMPLE_DELIVERY",
    "GitHubProvider",
    "ReviewComment",
    "ReviewContext",
    "ReviewDecision",
    "Verdict",
    "build_review_prompt",
    "collect_review_context",
    "condition_name_for",
    "events",
    "handshake",
    "parse",
    "parse_review_payload",
    "review_pr",
    "verify",
]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "00000000-0000-0000-0000-000000000000",
        "X-Hub-Signature-256": f"sha256={signature}",
    }


#: A real `pull_request.opened` delivery, trimmed to the fields the parser reads.
#: The conformance harness signs and replays it, so `verify` and `parse` are
#: checked against an actual payload rather than against each other.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"action": "opened", "number": 7,'
        b' "pull_request": {"number": 7, "title": "Add a feature",'
        b' "html_url": "https://github.com/octo/repo/pull/7", "updated_at": "2024-01-01T00:00:00Z"},'
        b' "repository": {"full_name": "octo/repo"}, "sender": {"login": "octocat"}}'
    ),
)
