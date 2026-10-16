"""Receive GitLab webhooks in Flyte, and see one arrive without leaving your laptop.

Two ways to run this. The second needs no GitLab account at all:

    python gitlab_webhooks.py --local   # replay a real sample delivery in-process
    python gitlab_webhooks.py           # deploy the receiver to Flyte

`--local` runs the app through FastAPI's test client and posts this plugin's
`SAMPLE_DELIVERY` — a `merge_request` "open" delivery — carrying a throwaway
token. You see the delivery verified, normalized, and dispatched to a handler,
which is the whole path a real webhook takes.

To receive real events, deploy it and point GitLab at `<app-url>/webhook/gitlab`
from Project Settings -> Webhooks.

Setup for the real thing:
    flyte create secret GITLAB_WEBHOOK_TOKEN --value <token>

`<token>` is whatever you typed into the webhook's *Secret token* field — the
field you set in Project Settings -> Webhooks -> Add webhook. GitLab does not
sign its webhooks; it sends that token verbatim in `X-Gitlab-Token`, so a
mismatch shows up as a 401 in the project's webhook *Recent deliveries*.
"""

import os
import sys

import flyte
from flyte.extras.webhooks import WebhookAppEnvironment

from flyteplugins.gitlab import SAMPLE_DELIVERY, GitLabProvider, events, payloads

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-gitlab[app]")

app_env = WebhookAppEnvironment(
    name="gitlab-webhooks",
    providers=[GitLabProvider()],
    image=image,
)


@app_env.on_event(events.MergeRequest.OPEN)
async def on_primary(event):
    """React to the event this plugin's sample delivery carries.

    Returning a dict is enough to see the path working. To do real work, launch
    a deployed task instead — see `launch_a_task` below.
    `payloads.merge_request` is a typed view of `event.payload`, so GitLab's
    field names autocomplete instead of being remembered.
    """
    payload = payloads.merge_request(event)
    return {
        "saw": event.qualified_type,
        "resource": event.resource_id,
        "title": event.title,
        "source_branch": payload.get("object_attributes", {}).get("source_branch"),
        # The key `run_once` would dedupe on. Replaying the same delivery
        # produces the same key, which is what makes a redelivery a no-op.
        "dedupe_key": event.dedupe_key(),
    }


@app_env.on_event(events.Issue.OPEN)
async def on_secondary(event):
    """A second handler, to show dispatch picking the right one per event."""
    return {"saw": event.qualified_type, "resource": event.resource_id}


async def launch_a_task(event):
    """What a handler looks like once it does real work.

    Not registered above, because it needs `gitlab-triage.triage_mr` deployed first
    and a Flyte backend to launch into. Wire it up with:

        @app_env.on_event(events.MergeRequest.OPEN)

    `run_once` returns the run already carrying the dedupe key when one is
    live or has succeeded, rather than launching a second, so GitLab redelivering an event — which
    it does on any non-2xx — never starts a second run.
    """
    import flyte.remote as remote
    from flyte.extras.webhooks import run_once

    task = remote.Task.get(name="gitlab-triage.triage_mr", auto_version="latest")
    # Always `.aio`: the blocking form stalls the app's event loop, and
    # webhook senders time deliveries out in seconds.
    result = await run_once.aio(task, key=event.dedupe_key(), project=event.scope)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


def _try_locally() -> None:
    """Post this plugin's sample delivery to the app, in-process."""
    from fastapi.testclient import TestClient

    secret = os.environ.setdefault(GitLabProvider.default_secret_env, "local-trial-secret")
    build_headers, body = SAMPLE_DELIVERY
    client = TestClient(app_env.app)

    print("POST /webhook/gitlab  (with a throwaway token)")
    response = client.post("/webhook/gitlab", content=body, headers=build_headers(body, secret))
    print(f"  {response.status_code}  {response.json()}\n")

    print("the same delivery again — note the identical dedupe_key:")
    again = client.post("/webhook/gitlab", content=body, headers=build_headers(body, secret))
    print(f"  {again.status_code}  {again.json()}\n")

    print("a delivery with the wrong token is refused:")
    bad = client.post("/webhook/gitlab", content=body, headers=build_headers(body, "wrong-token"))
    print(f"  {bad.status_code}  {bad.json()}\n")

    print("normalized events the app has seen:")
    for seen in client.get("/api/events").json():
        print(f"  {seen['provider']}  {seen['qualified_type']}  resource={seen['resource_id']}")


if __name__ == "__main__":
    if "--local" in sys.argv:
        _try_locally()
    else:
        flyte.init_from_config()
        handle = flyte.serve(app_env)
        handle.activate(wait=True)
        print(f"Dashboard ready at {handle.endpoint}")
        print(f"Point GitLab at {handle.endpoint}/webhook/gitlab")
