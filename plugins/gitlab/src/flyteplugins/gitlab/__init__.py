"""GitLab webhooks for Flyte.

Hand a `GitLabProvider()` to a `WebhookAppEnvironment` and register handlers with the
typed constants in `events`:

```python
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, run_once
from flyteplugins.gitlab import GitLabProvider, events

# GitLabProvider.default_secret_env (GITLAB_WEBHOOK_TOKEN) is mounted for you.
app_env = WebhookAppEnvironment(name="gitlab-webhooks", providers=[GitLabProvider()])


@app_env.on_event(events.MergeRequest.OPEN)
async def triage(event):
    import flyte.remote as remote

    task = remote.Task.get(name="gitlab-triage.triage_mr", auto_version="latest")
    result = await run_once.aio(task, key=event.dedupe_key(), repo=event.scope)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}
```

Note GitLab does not sign its webhooks; it authenticates with a shared token
sent verbatim in `X-Gitlab-Token` (the legacy *Secret token* mode — its newer
Standard-Webhooks *Signing token* mode, with `webhook-signature` /
`webhook-id` / `webhook-timestamp` headers, is not covered). See `_provider`
for what this plugin does with that. Calling the GitLab API (opening MRs, commenting, approving) is not
this plugin's job — use the `python-gitlab` package from your tasks, installed
via `flyteplugins-gitlab[gitlab]`. See `examples/external_saas_integrations`.
"""

from . import events, payloads
from ._provider import GitLabProvider, parse, verify

__all__ = [
    "SAMPLE_DELIVERY",
    "GitLabProvider",
    "events",
    "parse",
    "payloads",
    "verify",
]


def _sample_headers(body: bytes, secret: str) -> dict[str, str]:
    # No signature to compute: GitLab sends a static shared token.
    return {
        "X-Gitlab-Event": "Merge Request Hook",
        "X-Gitlab-Event-UUID": "00000000-0000-0000-0000-000000000000",
        "X-Gitlab-Token": secret,
    }


#: A real `merge_request` "open" delivery, trimmed to the fields the parser reads.
#: The conformance harness replays it under the shared token, so `verify` and
#: `parse` are checked against an actual payload rather than against each other.
SAMPLE_DELIVERY = (
    _sample_headers,
    (
        b'{"object_kind": "merge_request", "event_type": "merge_request",'
        b' "user": {"name": "Octocat", "username": "octocat"},'
        b' "project": {"id": 1, "name": "repo", "path_with_namespace": "octo/repo"},'
        b' "object_attributes": {"id": 99, "iid": 7, "action": "open", "state": "opened",'
        b' "title": "Add a feature", "url": "https://gitlab.com/octo/repo/-/merge_requests/7",'
        b' "source_branch": "feature", "target_branch": "main",'
        b' "updated_at": "2024-01-01T00:00:00Z"}}'
    ),
)
