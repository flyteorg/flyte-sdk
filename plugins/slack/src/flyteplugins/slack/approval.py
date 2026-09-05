"""Slack-native approvals: post buttons, pause the run, resume on the click.

The task half posts a Block Kit message and waits on a `flyteplugins-hitl`
event; the webhook half answers that event when a button is clicked. Together
they make "deploy to prod?" a one-line await:

```python
# in a task (needs flyteplugins-hitl installed: flyteplugins-slack[approval])
from flyteplugins.slack import approval

decision = await approval.request.aio("C0DEPLOYS", "Deploy release-42 to prod?")
if decision == "approve":
    ...
```

```python
# in the webhook app
from flyteplugins.slack import SlackProvider, approval

app_env = WebhookAppEnvironment(name="webhooks", providers=[SlackProvider()])
approval.register(app_env)
```

The button's `value` carries the hitl request id and response path, so the
webhook app needs no configuration to answer — it writes the response where
the waiting task already polls, exactly as the hitl form app would, then
replaces the buttons with a "decided by" line so nobody clicks twice.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Sequence

from flyte.syncify import syncify

from . import notify
from .events import Interaction

if TYPE_CHECKING:
    from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent

#: Every approval button's action_id starts with this; the option follows the
#: colon, since Slack requires distinct action_ids within one block.
ACTION_PREFIX = "hitl-decision"


def blocks(
    prompt: str,
    options: Sequence[str],
    *,
    request_id: str,
    response_path: str,
) -> list[dict[str, Any]]:
    """The approval message: the prompt, then one button per option.

    Exposed separately so callers who want richer messages — context blocks,
    fields, images — can embed the buttons in their own layout and still be
    answered by `register`'s handler.
    """
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": prompt}},
        {
            "type": "actions",
            "block_id": ACTION_PREFIX,
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": option},
                    "action_id": f"{ACTION_PREFIX}:{option}",
                    "value": json.dumps({"request_id": request_id, "response_path": response_path, "choice": option}),
                    **({"style": "primary"} if index == 0 else {}),
                }
                for index, option in enumerate(options)
            ],
        },
    ]


@syncify
async def request(
    channel: str,
    prompt: str,
    *,
    options: Sequence[str] = ("approve", "reject"),
    thread_ts: str | None = None,
    timeout_seconds: int = 3600,
    name: str = "slack-approval",
    token: str | None = None,
) -> str:
    """Post an approval message to `channel` and pause until a button is clicked.

    Returns the chosen option. Runs inside a deployed task only: the hitl
    event it waits on serves its own app and polls object storage. Use
    `.aio(...)` from async tasks and the bare call from sync ones. Raises
    `TimeoutError` when nobody decides within `timeout_seconds`.
    """
    try:
        from flyteplugins.hitl import new_event
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "flyteplugins-hitl is not installed. Install 'flyteplugins-slack[approval]' to use approvals."
        ) from exc

    event = await new_event.aio(name, data_type=str, prompt=prompt, timeout_seconds=timeout_seconds)
    await notify.post(
        channel,
        prompt,
        blocks=blocks(prompt, options, request_id=event.request_id, response_path=event.response_path),
        thread_ts=thread_ts,
        token=token,
    )
    return await event.wait.aio()


async def _ensure_initialized() -> None:
    """Initialize flyte lazily, so registering the handler costs nothing at import.

    `current_project` is a cheap read once initialized, so this needs no guard
    flag of its own.
    """
    import flyte
    import flyte.errors

    try:
        flyte.current_project()
    except flyte.errors.InitializationError:
        await flyte.init_in_cluster.aio()


async def _answer(request_id: str, response_path: str, choice: str) -> None:
    """Write the response where the waiting task polls — what the hitl form app does."""
    import flyte.storage as storage

    await _ensure_initialized()
    response = json.dumps(
        {"value": choice, "status": "completed", "request_id": request_id, "data_type": "str"}
    ).encode()
    await storage.put_stream(response, to_path=response_path)


async def _on_decision(event: WebhookEvent) -> dict[str, Any] | None:
    """Answer the hitl event a clicked approval button names, then retire the buttons."""
    action = (event.payload.get("actions") or [{}])[0]
    if not str(action.get("action_id", "")).startswith(ACTION_PREFIX):
        return None  # someone else's button; stay out of the response envelope
    data = json.loads(action["value"])
    await _answer(data["request_id"], data["response_path"], data["choice"])
    response_url = event.payload.get("response_url")
    if response_url:
        decided_by = f" — decided by <@{event.actor}>" if event.actor else ""
        await notify.respond(response_url, text=f"*{data['choice']}*{decided_by}", replace_original=True)
    return {"request_id": data["request_id"], "choice": data["choice"]}


def register(app_env: WebhookAppEnvironment) -> None:
    """Register the decision handler; every approval button is answered from then on."""
    app_env.on_event(Interaction.BLOCK_ACTIONS)(_on_decision)
