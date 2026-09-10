"""Slack-native approvals: post buttons, pause the run, resume on the click.

The task half posts a Block Kit message and parks the run on a
`flyte.new_condition`; the webhook half signals that condition when a button is
clicked. Together they make "deploy to prod?" a one-line await:

```python
# in a task
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

The button's `value` carries the run, action, and condition names, so the
webhook app needs no configuration to answer — it looks the condition up with
`flyte.remote.Condition.get` and signals it — then replaces the buttons with a
"decided by" line so nobody clicks twice.

A condition is also answerable from the Flyte UI, so an approval that never
gets clicked in Slack is not stuck: the run shows the same prompt, and either
path resolves it.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Sequence

from flyte.syncify import syncify

from . import notify, payloads
from .events import Interaction

if TYPE_CHECKING:
    from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent

#: Every approval button's action_id starts with this; the option follows the
#: colon, since Slack requires distinct action_ids within one block.
ACTION_PREFIX = "flyte-condition"


def blocks(
    prompt: str,
    options: Sequence[str],
    *,
    condition: str,
    run_name: str,
    action_name: str,
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
                    "value": json.dumps(
                        {"condition": condition, "run": run_name, "action": action_name, "choice": option}
                    ),
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
    timeout: timedelta | int | float | None = 3600,
    name: str = "slack-approval",
    token: str | None = None,
) -> str:
    """Post an approval message to `channel` and park the run until a button is clicked.

    Returns the chosen option. Runs inside a task only — the condition is
    registered against the running action. Use `.aio(...)` from async tasks and
    the bare call from sync ones.

    Args:
        channel: Channel id to post the approval message to.
        prompt: The question. Shown in Slack and as the condition's prompt in
            the Flyte UI, so a reviewer answering there sees the same text.
        options: One button per option; the first is styled primary. The
            chosen option is what this returns.
        thread_ts: Post into an existing thread rather than the channel root.
        timeout: Forwarded to `flyte.new_condition`. On expiry `wait()` raises
            `flyte.errors.ConditionTimedoutError`.
        name: Condition name, and how the click finds it. Give concurrent
            approvals in one run distinct names.
        token: Explicit bot token; otherwise `SLACK_BOT_TOKEN`.
    """
    import flyte

    condition = await flyte.new_condition.aio(
        name,
        prompt=prompt,
        prompt_type="markdown",
        data_type=str,
        timeout=timeout,
    )
    # task_action, not action: @trace swaps `action` for a pseudo-action, but the
    # condition is registered under the real running task, which is what
    # `Condition.get(action_name=...)` filters on.
    tctx = flyte.ctx()
    action = tctx.task_action or tctx.action
    await notify.post(
        channel,
        prompt,
        blocks=blocks(
            prompt,
            options,
            condition=name,
            run_name=action.run_name or action.name,
            action_name=action.name,
        ),
        thread_ts=thread_ts,
        token=token,
    )
    return await condition.wait.aio()


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


async def _answer(condition: str, run_name: str, action_name: str, choice: str) -> None:
    """Signal the parked condition, resuming the run that posted the buttons."""
    import flyte.remote as remote

    await _ensure_initialized()
    found = await remote.Condition.get.aio(condition, run_name=run_name, action_name=action_name)
    if found is None:
        raise RuntimeError(
            f"condition {condition!r} not found on run {run_name!r} (action {action_name!r}); "
            "it may have already been signaled, timed out, or the run may have ended"
        )
    await found.signal.aio(choice)


async def _on_decision(event: WebhookEvent) -> dict[str, Any] | None:
    """Signal the condition a clicked approval button names, then retire the buttons."""
    payload = payloads.block_actions(event)
    action = (payload.get("actions") or [{}])[0]
    if not str(action.get("action_id", "")).startswith(ACTION_PREFIX):
        return None  # someone else's button; stay out of the response envelope
    data = json.loads(action["value"])
    await _answer(data["condition"], data["run"], data["action"], data["choice"])
    response_url = payload.get("response_url")
    if response_url:
        decided_by = f" — decided by <@{event.actor}>" if event.actor else ""
        await notify.respond(response_url, text=f"*{data['choice']}*{decided_by}", replace_original=True)
    return {"condition": data["condition"], "run": data["run"], "choice": data["choice"]}


def register(app_env: WebhookAppEnvironment) -> None:
    """Register the decision handler; every approval button is answered from then on."""
    app_env.on_event(Interaction.BLOCK_ACTIONS)(_on_decision)
