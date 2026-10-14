"""Send Slack messages from tasks: post, update, delete, respond.

Receiving webhooks is the app's job; sending is your task's. These helpers
cover the sends every integration ends up hand-rolling — Nodey's
`send_notification` task is fifty lines of `requests` and `ok` checking that
`post()` replaces:

```python
from flyteplugins.slack import notify

ts = await notify.post("C0DEPLOYS", "deploy started", thread_ts=thread_ts)
await notify.update("C0DEPLOYS", ts, "deploy finished")
```

`post`/`update`/`delete` call the Slack Web API with a bot token, read from
`SLACK_BOT_TOKEN` — mount it with `flyte.Secret("SLACK_BOT_TOKEN")`. That is
the `xoxb-` credential from *OAuth & Permissions*, not the signing secret the
webhook receiver verifies with.

`respond()` needs no token at all: it posts to the `response_url` every
interaction and slash command carries (valid for 30 minutes, five uses), which
makes it the zero-setup way for a launched task to answer the click that
launched it.

A ready-made task environment ships at the bottom, so
`flyte deploy plugins/slack/src/flyteplugins/slack/notify.py` (or
`flyte.deploy(notify.env)`) gives other runs a `send` task to call.
"""

from __future__ import annotations

import os
from typing import Any

import flyte
import httpx

#: The bot token's environment variable. Mount it with `flyte.Secret("SLACK_BOT_TOKEN")`.
TOKEN_ENV = "SLACK_BOT_TOKEN"

_API_BASE = "https://slack.com/api"
_TIMEOUT_SECONDS = 15.0

#: Swapped for an `httpx.MockTransport` in tests; None means the real network.
_transport: httpx.AsyncBaseTransport | None = None


class SlackApiError(RuntimeError):
    """Slack answered but refused — `error` is Slack's code, e.g. `channel_not_found`.

    The two codes worth knowing: `not_in_channel` means the bot needs a
    `/invite`, and `missing_scope` names the OAuth scope to add (posting needs
    `chat:write`).
    """

    def __init__(self, method: str, error: str, response: dict[str, Any]) -> None:
        super().__init__(f"Slack API {method} failed: {error}")
        self.method = method
        self.error = error
        self.response = response


def _token(token: str | None) -> str:
    resolved = token or os.environ.get(TOKEN_ENV)
    if not resolved:
        raise RuntimeError(
            f"no Slack bot token: pass token= or mount flyte.Secret({TOKEN_ENV!r}) on the task environment. "
            "The bot token is the xoxb- credential under OAuth & Permissions at api.slack.com/apps."
        )
    return resolved


async def _call(method: str, payload: dict[str, Any], token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=_transport) as client:
        response = await client.post(
            f"{_API_BASE}/{method}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    response.raise_for_status()
    data = response.json()
    # Slack reports failure in the body, not the status code.
    if not data.get("ok"):
        raise SlackApiError(method, data.get("error", "unknown_error"), data)
    return data


async def post(
    channel: str,
    text: str | None = None,
    *,
    blocks: list[dict[str, Any]] | None = None,
    thread_ts: str | None = None,
    token: str | None = None,
) -> str:
    """Post a message; returns its `ts`, which threads replies and addresses `update`."""
    payload: dict[str, Any] = {"channel": channel}
    if text is not None:
        payload["text"] = text
    if blocks is not None:
        payload["blocks"] = blocks
    if thread_ts is not None:
        payload["thread_ts"] = thread_ts
    data = await _call("chat.postMessage", payload, _token(token))
    return data["ts"]


async def update(
    channel: str,
    ts: str,
    text: str | None = None,
    *,
    blocks: list[dict[str, Any]] | None = None,
    token: str | None = None,
) -> str:
    """Edit a posted message in place — progress counters, final status."""
    payload: dict[str, Any] = {"channel": channel, "ts": ts}
    if text is not None:
        payload["text"] = text
    if blocks is not None:
        payload["blocks"] = blocks
    data = await _call("chat.update", payload, _token(token))
    return data["ts"]


async def delete(channel: str, ts: str, *, token: str | None = None) -> None:
    """Delete one of the bot's own messages — the retract-a-misclick shape."""
    await _call("chat.delete", {"channel": channel, "ts": ts}, _token(token))


async def respond(
    response_url: str,
    text: str | None = None,
    *,
    blocks: list[dict[str, Any]] | None = None,
    replace_original: bool | None = None,
    response_type: str | None = None,
) -> None:
    """Answer an interaction or slash command via its `response_url` — no token needed.

    Args:
        response_url: From `event.payload["response_url"]`. Slack honors it for
            30 minutes, five times.
        replace_original: True swaps the message the button lives on for this
            one — how approval buttons become an "approved by" line.
        response_type: `"in_channel"` to make a slash-command reply visible to
            everyone; Slack's default is ephemeral.
    """
    payload: dict[str, Any] = {}
    if text is not None:
        payload["text"] = text
    if blocks is not None:
        payload["blocks"] = blocks
    if replace_original is not None:
        payload["replace_original"] = replace_original
    if response_type is not None:
        payload["response_type"] = response_type
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=_transport) as client:
        response = await client.post(response_url, json=payload)
    response.raise_for_status()


# ----------------------------------------------------------------------
# a deployable notification task, for runs that should not carry the token
# ----------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="slack-notify",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    secrets=[flyte.Secret(key=TOKEN_ENV, as_env_var=TOKEN_ENV)],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def send(channel: str, text: str, thread_ts: str | None = None) -> str:
    """Post `text` to `channel`, threading under `thread_ts` when given.

    Deploy this environment once and only it holds the bot token; every other
    run posts through `flyte.run(notify.send, ...)` without mounting a secret.
    """
    return await post(channel, text, thread_ts=thread_ts)


if __name__ == "__main__":
    flyte.init_from_config()
    for deployment in flyte.deploy(env):
        print(deployment.table_repr())
