"""Slack verification and payload normalization.

Slack sends three body shapes to an app, all signed the same way:

* Events API callbacks — JSON;
* interactivity payloads (Block Kit actions, shortcuts, modals) — form-encoded,
  the JSON under a `payload` field;
* slash commands — form-encoded fields.

`verify` covers all three, since the v0 HMAC signs the raw body regardless of
encoding. `parse` normalizes each into a `WebhookEvent`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any, ClassVar, Mapping

from flyte.extras.webhooks import (
    Provider,
    SignatureError,
    WebhookEvent,
    constant_time_equals,
    json_body,
    lower_headers,
)

#: Reject requests whose timestamp is older than this (replay protection).
MAX_REQUEST_AGE_SECONDS = 60 * 5


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Slack-Signature` v0 HMAC, within the replay window."""
    lowered = lower_headers(headers)
    timestamp, signature = lowered.get("x-slack-request-timestamp"), lowered.get("x-slack-signature")
    if not timestamp or not signature or not signature.startswith("v0="):
        return False
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - sent_at) > MAX_REQUEST_AGE_SECONDS:
        return False
    # Sign the raw bytes and the raw header. Decoding the body and re-encoding it
    # would corrupt any byte Slack signed but Python cannot decode, and running
    # the timestamp through int() would drop whatever formatting Slack signed.
    basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return constant_time_equals(expected, signature)


def _form_fields(body: bytes) -> dict[str, str] | None:
    """Decode Slack's form-encoded deliveries, or None when the body is JSON.

    Interactivity payloads and slash commands arrive as
    `application/x-www-form-urlencoded`; the Events API arrives as JSON.
    Sniffing the body rather than trusting Content-Type keeps `parse` a pure
    function of the payload, which is what the conformance harness replays.
    """
    if body[:1] in (b"{", b"["):
        return None
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "=" not in decoded.split("&", 1)[0]:
        return None
    return {key: values[0] for key, values in urllib.parse.parse_qs(decoded, keep_blank_values=True).items()}


def handshake(headers: Mapping[str, str], body: bytes) -> dict[str, Any] | None:
    """Answer Slack's reachability probes, sent before events flow.

    Two exist: the `url_verification` challenge on the Events API URL, and the
    form-encoded `ssl_check` probe on interactivity and slash-command URLs.
    Ordinary deliveries return None and proceed to verification.
    """
    form = _form_fields(body)
    if form is not None:
        return {"ok": True} if form.get("ssl_check") else None
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("type") == "url_verification":
        return {"challenge": str(data.get("challenge", ""))}
    return None


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize any Slack delivery — event callback, interaction, or slash command."""
    form = _form_fields(body)
    if form is not None:
        if "payload" in form:
            return _parse_interaction(form["payload"])
        if "command" in form:
            return _parse_command(form)
        raise SignatureError("form-encoded delivery carries neither `payload` nor `command`")

    payload = json_body(body)
    event = payload.get("event") or {}
    if not isinstance(event, dict) or not event:
        raise SignatureError("event payload is missing its `event` object")
    channel, ts = event.get("channel"), event.get("ts")
    return WebhookEvent(
        provider="slack",
        event_type=event.get("type", "unknown"),
        action=event.get("subtype"),
        delivery_id=payload.get("event_id", ""),
        # Keyed per message. Collapse a whole thread onto one run by passing
        # `event.payload["event"]["thread_ts"]` as your own key instead.
        resource_id=f"{channel}:{ts}" if channel and ts else None,
        scope=channel,
        title=(event.get("text") or "")[:120] or None,
        actor=event.get("user"),
        payload=payload,
    )


def _parse_interaction(raw: str) -> WebhookEvent:
    """Normalize an interactivity payload (`block_actions`, shortcuts, modals).

    The action is the `action_id` (or `callback_id`), so one button registers
    as `"block_actions.<action_id>"`. `payload` carries Slack's full JSON —
    `actions`, `container`, `message`, `response_url` — for the handler.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SignatureError(f"interaction `payload` is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SignatureError("interaction `payload` is not a JSON object")

    interaction_type = payload.get("type", "unknown")
    action = (payload.get("actions") or [{}])[0]
    view = payload.get("view") or {}
    action_id = action.get("action_id") or payload.get("callback_id") or view.get("callback_id")
    container = payload.get("container") or {}
    channel = (payload.get("channel") or {}).get("id") or container.get("channel_id")
    ts = container.get("message_ts") or (payload.get("message") or {}).get("ts")
    return WebhookEvent(
        provider="slack",
        event_type=interaction_type,
        action=action_id,
        # trigger_id is unique per interaction, so interactions with no message
        # to key on (modals, global shortcuts) still dedupe per delivery.
        delivery_id=payload.get("trigger_id", ""),
        resource_id=f"{channel}:{ts}" if channel and ts else None,
        # action_ts is unique per click, so two clicks of one button on one
        # message get their own dedupe keys while a redelivery of either does not.
        occurred_at=action.get("action_ts") or payload.get("action_ts"),
        scope=channel,
        title=action_id or interaction_type,
        actor=(payload.get("user") or {}).get("id"),
        payload=payload,
    )


def _parse_command(form: Mapping[str, str]) -> WebhookEvent:
    """Normalize a slash command, so `/hi-agent` registers as `"command.hi-agent"`."""
    command, text = form.get("command", ""), form.get("text", "")
    return WebhookEvent(
        provider="slack",
        event_type="command",
        action=command.lstrip("/") or None,
        # No resource: every invocation is its own delivery, keyed by trigger_id.
        delivery_id=form.get("trigger_id", ""),
        scope=form.get("channel_id"),
        title=f"{command} {text}".strip()[:120] or None,
        actor=form.get("user_id"),
        payload=dict(form),
    )


class SlackProvider(Provider):
    """Slack's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.slack import SlackProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[SlackProvider()])
    ```

    One route serves all three of Slack's delivery shapes, so point Event
    Subscriptions, Interactivity & Shortcuts, and every slash command at the
    same `/webhook/slack` URL.

    `WebhookAppEnvironment` mounts `default_secret_env` for you, so it does not
    need naming again in `secrets=`.

    Args:
        secret_env: Environment variable holding the secret. Pass one only to
            point this provider at a secret stored under a different name;
            otherwise `default_secret_env` applies.
    """

    default_secret_env: ClassVar[str] = "SLACK_SIGNING_SECRET"

    def __init__(self, *, secret_env: str | None = None) -> None:
        super().__init__(
            name="slack",
            secret_env=secret_env or self.default_secret_env,
            verify=verify,
            parse=parse,
            handshake=handshake,
            setup_hint="api.slack.com/apps -> Event Subscriptions",
        )
