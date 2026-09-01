"""Per-provider verification and payload normalization.

Everything a provider needs to be supported lives in one `Provider` here: the
header its signature arrives in, how to verify it, and how to turn its payload
into a `WebhookEvent`. Adding a sixth product means adding one entry, not
another package.

Every comparison is done on bytes. `hmac.compare_digest` raises `TypeError` on
`str` operands containing non-ASCII, and these headers are attacker-controlled —
ASGI servers hand Starlette raw header bytes, which it decodes as latin-1, so a
crafted header would otherwise turn a clean 401 into a 500.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ._errors import SignatureError
from ._event import WebhookEvent

#: Reject Slack requests whose timestamp is older than this (replay protection).
SLACK_MAX_REQUEST_AGE_SECONDS = 60 * 5


def _json_body(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignatureError(f"invalid webhook body: {exc}") from exc
    return data if isinstance(data, dict) else {"data": data}


def _eq(a: str, b: str) -> bool:
    """Constant-time compare on bytes, never raising on non-ASCII input."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _hex_hmac(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


# ----------------------------------------------------------------------
# github
# ----------------------------------------------------------------------


def _verify_github(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    signature = _lower(headers).get("x-hub-signature-256")
    if not signature or not signature.startswith("sha256="):
        return False
    return _eq(_hex_hmac(secret, body), signature.removeprefix("sha256="))


def _parse_github(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    lowered = _lower(headers)
    event_type = lowered.get("x-github-event")
    if not event_type:
        raise SignatureError("missing X-GitHub-Event header")
    payload = _json_body(body)

    repo = payload.get("repository") or {}
    issue_or_pr = payload.get("pull_request") or payload.get("issue") or {}
    # `comment` covers issue_comment / commit_comment / review comments; `review`
    # covers pull_request_review. Either identifies the event within its issue,
    # so two comments on one issue do not collapse onto a single dedupe key.
    comment = payload.get("comment") or payload.get("review") or {}
    number = issue_or_pr.get("number")

    resource = None
    if repo.get("full_name") and number is not None:
        resource = f"{repo['full_name']}#{number}"
        if comment.get("id") is not None:
            resource = f"{resource}:{comment['id']}"

    return WebhookEvent(
        provider="github",
        event_type=event_type,
        action=payload.get("action"),
        delivery_id=lowered.get("x-github-delivery", ""),
        resource_id=resource,
        occurred_at=issue_or_pr.get("updated_at"),
        scope=repo.get("full_name"),
        title=issue_or_pr.get("title"),
        url=comment.get("html_url") or issue_or_pr.get("html_url"),
        actor=(payload.get("sender") or {}).get("login"),
        payload=payload,
    )


# ----------------------------------------------------------------------
# slack
# ----------------------------------------------------------------------


def _verify_slack(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    lowered = _lower(headers)
    timestamp, signature = lowered.get("x-slack-request-timestamp"), lowered.get("x-slack-signature")
    if not timestamp or not signature or not signature.startswith("v0="):
        return False
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - sent_at) > SLACK_MAX_REQUEST_AGE_SECONDS:
        return False
    # Sign the raw bytes and the raw header. Decoding the body and re-encoding it
    # would corrupt any byte Slack signed but Python cannot decode, and running
    # the timestamp through int() would drop whatever formatting Slack signed.
    basestring = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return _eq(expected, signature)


def _parse_slack(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    payload = _json_body(body)
    event = payload.get("event") or {}
    if not isinstance(event, dict) or not event:
        raise SignatureError("event payload is missing its `event` object")
    channel, ts = event.get("channel"), event.get("ts")
    return WebhookEvent(
        provider="slack",
        event_type=event.get("type", "unknown"),
        action=event.get("subtype"),
        delivery_id=payload.get("event_id", ""),
        # Keyed per message. Collapse a thread onto one run by passing
        # `event.payload["event"]["thread_ts"]` as your own key instead.
        resource_id=f"{channel}:{ts}" if channel and ts else None,
        scope=channel,
        title=(event.get("text") or "")[:120] or None,
        actor=event.get("user"),
        payload=payload,
    )


def slack_url_verification(body: bytes) -> str | None:
    """Return the `challenge` when this is Slack's one-off handshake, else None."""
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("type") == "url_verification":
        return str(data.get("challenge", ""))
    return None


# ----------------------------------------------------------------------
# linear
# ----------------------------------------------------------------------


def _verify_linear(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    signature = _lower(headers).get("x-linear-signature")
    return bool(signature) and _eq(_hex_hmac(secret, body), signature.strip())


def _linear_team_id(data: dict[str, Any]) -> str | None:
    """Find the team id, which Comment and Reaction payloads nest on the issue."""
    issue = data.get("issue") or {}
    for candidate in (
        data.get("teamId"),
        (data.get("team") or {}).get("id"),
        issue.get("teamId"),
        (issue.get("team") or {}).get("id"),
    ):
        if candidate:
            return str(candidate)
    return None


def _parse_linear(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    payload = _json_body(body)
    data = payload.get("data") or {}
    return WebhookEvent(
        provider="linear",
        event_type=payload.get("type", "Unknown"),
        action=payload.get("action", "unknown"),
        delivery_id=str(payload.get("webhookId") or ""),
        resource_id=data.get("id"),
        # `updatedAt` is on the entity; `createdAt` is the delivery time and the
        # only timestamp on payloads whose entity carries none.
        occurred_at=data.get("updatedAt") or payload.get("createdAt"),
        scope=_linear_team_id(data),
        title=data.get("title"),
        url=data.get("url") or payload.get("url"),
        actor=(data.get("creator") or {}).get("name"),
        payload=payload,
    )


# ----------------------------------------------------------------------
# clickup
# ----------------------------------------------------------------------


def _verify_clickup(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    signature = _lower(headers).get("x-clickup-signature")
    return bool(signature) and _eq(_hex_hmac(secret, body), signature.strip())


def _parse_clickup(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    payload = _json_body(body)
    task = payload.get("task") or {}
    # ClickUp puts the list id at the top level on list-scoped events and only on
    # the nested task for task-scoped ones.
    list_id = payload.get("list_id") or (task.get("list") or {}).get("id")
    task_id = payload.get("task_id") or task.get("id")
    return WebhookEvent(
        provider="clickup",
        event_type=payload.get("event", "unknown"),
        delivery_id=str(payload.get("webhook_id") or ""),
        resource_id=str(task_id) if task_id is not None else None,
        occurred_at=str(payload.get("timestamp")) if payload.get("timestamp") is not None else None,
        scope=str(list_id) if list_id is not None else None,
        title=task.get("name"),
        url=task.get("url"),
        payload=payload,
    )


# ----------------------------------------------------------------------
# jira
# ----------------------------------------------------------------------


def _verify_jira(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Jira Cloud does not sign its webhooks.

    The receiver falls back to a shared token in `X-Webhook-Token`, which
    whatever sits in front of the app has to inject — Jira itself cannot send
    custom headers.
    """
    token = _lower(headers).get("x-webhook-token")
    return bool(token) and _eq(token.strip(), secret)


def _parse_jira(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    payload = _json_body(body)
    issue = payload.get("issue") or {}
    fields = issue.get("fields") or {}
    user = payload.get("user") or {}
    return WebhookEvent(
        provider="jira",
        event_type=payload.get("webhookEvent", "unknown"),
        resource_id=issue.get("key"),
        occurred_at=str(payload.get("timestamp")) if payload.get("timestamp") is not None else None,
        scope=(fields.get("project") or {}).get("key"),
        title=fields.get("summary"),
        actor=user.get("displayName") or user.get("name"),
        payload=payload,
    )


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    """Everything the receiver needs to accept one product's webhooks.

    Args:
        name: URL segment and `WebhookEvent.provider` value.
        secret_env: Environment variable holding the signing secret or shared
            token, mounted from a `flyte.Secret`.
        verify: Returns True when a delivery is authentic.
        parse: Turns a verified delivery into a `WebhookEvent`.
        signed: False when the provider does not sign at all, so the dashboard
            can say so rather than implying a guarantee that is not there.
        setup_url: Where to configure the webhook, shown on the dashboard.
    """

    name: str
    secret_env: str
    verify: Callable[[bytes, Mapping[str, str], str], bool]
    parse: Callable[[Mapping[str, str], bytes], WebhookEvent]
    signed: bool = True
    setup_url: str = ""


GITHUB = Provider(
    name="github",
    secret_env="GITHUB_WEBHOOK_SECRET",
    verify=_verify_github,
    parse=_parse_github,
    setup_url="repository Settings -> Webhooks -> Add webhook",
)

SLACK = Provider(
    name="slack",
    secret_env="SLACK_SIGNING_SECRET",
    verify=_verify_slack,
    parse=_parse_slack,
    setup_url="api.slack.com/apps -> Event Subscriptions",
)

LINEAR = Provider(
    name="linear",
    secret_env="LINEAR_WEBHOOK_SECRET",
    verify=_verify_linear,
    parse=_parse_linear,
    setup_url="Linear Settings -> API -> Webhooks",
)

CLICKUP = Provider(
    name="clickup",
    secret_env="CLICKUP_WEBHOOK_SECRET",
    verify=_verify_clickup,
    parse=_parse_clickup,
    setup_url="Space Settings -> Integrations -> Webhooks",
)

JIRA = Provider(
    name="jira",
    secret_env="JIRA_WEBHOOK_TOKEN",
    verify=_verify_jira,
    parse=_parse_jira,
    signed=False,
    setup_url="Jira Settings -> System -> Webhooks",
)

#: Every provider this plugin supports, by name.
PROVIDERS: dict[str, Provider] = {p.name: p for p in (GITHUB, SLACK, LINEAR, CLICKUP, JIRA)}
