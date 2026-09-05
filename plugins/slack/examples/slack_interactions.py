"""Handle Slack buttons and slash commands in Flyte, and try both without a Slack account.

Two ways to run this. The second needs no Slack account at all:

    python slack_interactions.py --local   # replay signed sample deliveries in-process
    python slack_interactions.py           # deploy the receiver to Flyte

Slack sends three delivery shapes, and one `/webhook/slack` route serves them
all: Events API callbacks (JSON — see `slack_webhooks.py`), interactivity
payloads (Block Kit buttons, shortcuts, modals), and slash commands. The last
two arrive form-encoded and are what this example demonstrates.

Setup for the real thing, at api.slack.com/apps, all pointing at the same URL:

    - Interactivity & Shortcuts -> Request URL: <app-url>/webhook/slack
    - Slash Commands -> create `/deploy` with Request URL: <app-url>/webhook/slack

    flyte create secret SLACK_SIGNING_SECRET --value <secret>

The signing secret is under *Basic Information* in your Slack app.
"""

import hashlib
import hmac
import json
import os
import sys
import time
from urllib.parse import urlencode

import flyte
from flyte.extras.webhooks import WebhookAppEnvironment

from flyteplugins.slack import SlackProvider, events

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("flyteplugins-slack[app]")

app_env = WebhookAppEnvironment(
    name="slack-interactions",
    providers=[SlackProvider()],
    image=image,
)


@app_env.on_event("block_actions.approve_deploy")
async def on_approval(event):
    """One button. A block action registers as `block_actions.<action_id>`.

    `event.payload` is Slack's full interaction JSON, so everything a
    slack_bolt `@app.action` handler reads from `body` is here: which message
    the button lives on (`container`), the clicked action's `value`
    (`actions`), and the `response_url` for posting a reply.
    """
    container = event.payload.get("container", {})
    action = event.payload["actions"][0]
    return {
        "approved_by": event.actor,
        "message": f"{container.get('channel_id')}:{container.get('message_ts')}",
        "value": action.get("value"),
        # Two clicks of one button are two events; a Slack redelivery of
        # either click is not. That is what makes `run_once` safe here.
        "dedupe_key": event.dedupe_key(),
    }


@app_env.on_event(events.Interaction.BLOCK_ACTIONS)
async def on_any_button(event):
    """Every block action, whatever its action_id — an audit-log shape."""
    return {"saw": event.qualified_type}


@app_env.on_event("command.deploy")
async def on_deploy_command(event):
    """One slash command. `/deploy` registers as `"command.deploy"`.

    Slash commands arrive as flat form fields, so `event.payload` is a dict of
    `command`, `text`, `channel_id`, `user_id`, `response_url`, ...

    Returning here answers Slack's HTTP POST, and Slack shows the user an
    error unless that happens within 3 seconds — so do nothing slower than
    `run_once.aio` and post progress back via `slack_sdk` from the launched
    task. See `launch_a_task` below for that shape.
    """
    return {
        "command": event.payload["command"],
        "args": event.payload.get("text", ""),
        "channel": event.scope,
        "requested_by": event.actor,
    }


async def launch_a_task(event):
    """What the command handler looks like once it does real work.

    Not registered above, because it needs `deployer.deploy` deployed first and
    a Flyte backend to launch into. Wire it up with:

        @app_env.on_event("command.deploy")
    """
    import flyte.remote as remote
    from flyte.extras.webhooks import run_once

    task = remote.Task.get(name="deployer.deploy", auto_version="latest")
    # Always `.aio`: the blocking form stalls the app's event loop, and Slack
    # times interactivity and command deliveries out in 3 seconds.
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        args=event.payload.get("text", ""),
        channel=event.scope,
        # The task posts progress here with slack_sdk once it is running.
        response_url=event.payload.get("response_url", ""),
    )
    if not result.created:
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name}


# ----------------------------------------------------------------------
# local trial: the deliveries Slack would send, signed and replayed in-process
# ----------------------------------------------------------------------


def _sign(body: bytes, secret: str) -> dict[str, str]:
    """Slack's v0 signature — the same scheme for all three delivery shapes."""
    timestamp = str(int(time.time()))
    base = b"v0:" + timestamp.encode() + b":" + body
    return {
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest(),
        "Content-Type": "application/x-www-form-urlencoded",
    }


#: A button click, as Slack delivers it: form-encoded, the JSON under `payload`.
BUTTON_CLICK = urlencode(
    {
        "payload": json.dumps(
            {
                "type": "block_actions",
                "trigger_id": "13345224609.738474920.8088930838d88f008e0",
                "user": {"id": "U0EMPLOYEE"},
                "channel": {"id": "C0DEPLOYS"},
                "container": {"type": "message", "channel_id": "C0DEPLOYS", "message_ts": "1700000000.000100"},
                "actions": [{"action_id": "approve_deploy", "value": "release-42", "action_ts": "1700000001.000000"}],
                "message": {"ts": "1700000000.000100", "text": "Deploy release-42 to prod?"},
                "response_url": "https://hooks.slack.com/actions/T0/123/abc",
            }
        )
    }
).encode()

#: A slash command invocation: flat form fields.
SLASH_COMMAND = urlencode(
    {
        "command": "/deploy",
        "text": "release-42 --canary",
        "channel_id": "C0DEPLOYS",
        "user_id": "U0EMPLOYEE",
        "trigger_id": "13345224609.738474921.9199041949e99f119f1",
        "response_url": "https://hooks.slack.com/commands/T0/456/def",
    }
).encode()


def _try_locally() -> None:
    """Post a button click and a slash command to the app, in-process."""
    from fastapi.testclient import TestClient

    secret = os.environ.setdefault(SlackProvider.default_secret_env, "local-trial-secret")
    client = TestClient(app_env.app)

    print("POST /webhook/slack  (a button click, signed)")
    response = client.post("/webhook/slack", content=BUTTON_CLICK, headers=_sign(BUTTON_CLICK, secret))
    print(f"  {response.status_code}  {response.json()}\n")

    print("POST /webhook/slack  (/deploy release-42 --canary, signed)")
    response = client.post("/webhook/slack", content=SLASH_COMMAND, headers=_sign(SLASH_COMMAND, secret))
    print(f"  {response.status_code}  {response.json()}\n")

    print("the ssl_check probe Slack sends when you save a Request URL:")
    probe = client.post(
        "/webhook/slack",
        content=b"token=abc&ssl_check=1",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    print(f"  {probe.status_code}  {probe.json()}\n")

    print("an unsigned click is refused:")
    bad = client.post("/webhook/slack", content=BUTTON_CLICK, headers={})
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
        print(f"Point Interactivity & Shortcuts and your slash commands at {handle.endpoint}/webhook/slack")
