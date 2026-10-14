"""Receive ClickUp webhooks in Flyte, and see one arrive without leaving your laptop.

Two ways to run this. The second needs no ClickUp account at all:

    python clickup_webhooks.py --local   # replay a real sample delivery in-process
    python clickup_webhooks.py           # deploy the receiver to Flyte

`--local` runs the app through FastAPI's test client and posts this plugin's
`SAMPLE_DELIVERY` — a `taskCreated` delivery — signed with a throwaway secret. You see the
delivery verified, normalized, and dispatched to a handler, which is the whole
path a real webhook takes.

To receive real events, deploy it and point ClickUp at `<app-url>/webhook/clickup`
from Space Settings -> Integrations -> Webhooks.

Setup for the real thing:
    flyte create secret CLICKUP_WEBHOOK_SECRET --value <secret>

ClickUp shows the signing secret once, when you create the webhook.
"""

import os
import sys

import flyte
from flyte.extras.webhooks import WebhookAppEnvironment

from flyteplugins.clickup import SAMPLE_DELIVERY, ClickUpProvider, events

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-clickup[app]")

app_env = WebhookAppEnvironment(
    name="clickup-webhooks",
    providers=[ClickUpProvider()],
    image=image,
)


@app_env.on_event(events.Task.CREATED)
async def on_primary(event):
    """React to the event this plugin's sample delivery carries.

    Returning a dict is enough to see the path working. To do real work, launch
    a deployed task instead — see `launch_a_task` below.
    """
    return {
        "saw": event.qualified_type,
        "resource": event.resource_id,
        "title": event.title,
        # The key `run_once` would dedupe on. Replaying the same delivery
        # produces the same key, which is what makes a redelivery a no-op.
        "dedupe_key": event.dedupe_key(),
    }


@app_env.on_event(events.Task.STATUS_UPDATED)
async def on_secondary(event):
    """A second handler, to show dispatch picking the right one per event."""
    return {"saw": event.qualified_type, "resource": event.resource_id}


async def launch_a_task(event):
    """What a handler looks like once it does real work.

    Not registered above, because it needs `clickup-tickets.triage_task` deployed first
    and a Flyte backend to launch into. Wire it up with:

        @app_env.on_event(events.Task.CREATED)

    `run_once` returns the run already carrying the dedupe key when one is
    live or has succeeded, rather than launching a second, so ClickUp redelivering an event — which
    it does on any non-2xx — never starts a second run.
    """
    import flyte.remote as remote
    from flyte.extras.webhooks import run_once

    task = remote.Task.get(name="clickup-tickets.triage_task", auto_version="latest")
    # Always `.aio`: the blocking form stalls the app's event loop, and
    # webhook senders time deliveries out in seconds.
    result = await run_once.aio(task, key=event.dedupe_key(), task_id=event.resource_id)
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


def _try_locally() -> None:
    """Post this plugin's sample delivery to the app, in-process."""
    from fastapi.testclient import TestClient

    secret = os.environ.setdefault(ClickUpProvider.default_secret_env, "local-trial-secret")
    build_headers, body = SAMPLE_DELIVERY
    client = TestClient(app_env.app)

    print("POST /webhook/clickup  (signed with a throwaway secret)")
    response = client.post("/webhook/clickup", content=body, headers=build_headers(body, secret))
    print(f"  {response.status_code}  {response.json()}\n")

    print("the same delivery again — note the identical dedupe_key:")
    again = client.post("/webhook/clickup", content=body, headers=build_headers(body, secret))
    print(f"  {again.status_code}  {again.json()}\n")

    print("an unsigned delivery is refused:")
    bad = client.post("/webhook/clickup", content=body, headers={})
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
        print(f"Point ClickUp at {handle.endpoint}/webhook/clickup")
