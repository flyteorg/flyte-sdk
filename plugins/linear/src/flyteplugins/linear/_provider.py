"""Linear webhook verification and payload normalization."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from flyte.extras.webhooks import (
    Provider,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Linear-Signature` HMAC over the raw body."""
    signature = lower_headers(headers).get("x-linear-signature")
    if not signature:
        return False
    return constant_time_equals(hex_hmac_sha256(secret, body), signature.strip())


def _team_id(data: dict[str, Any]) -> str | None:
    """Find the team id, which Comment and Reaction payloads nest on the issue.

    Without these fallbacks a `scopes` allowlist drops every non-Issue event as
    unattributable.
    """
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


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a Linear delivery into a `WebhookEvent`."""
    payload = json_body(body)
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
        scope=_team_id(data),
        title=data.get("title"),
        url=data.get("url") or payload.get("url"),
        actor=(data.get("creator") or {}).get("name"),
        payload=payload,
    )


class LinearProvider(Provider):
    """Linear's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.linear import LinearProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[LinearProvider()])
    ```

    `WebhookAppEnvironment` mounts `default_secret_env` for you, so it does not
    need naming again in `secrets=`.

    Args:
        secret_env: Environment variable holding the secret. Pass one only to
            point this provider at a secret stored under a different name;
            otherwise `default_secret_env` applies.
    """

    default_secret_env: ClassVar[str] = "LINEAR_WEBHOOK_SECRET"

    def __init__(self, *, secret_env: str | None = None) -> None:
        super().__init__(
            name="linear",
            secret_env=secret_env or self.default_secret_env,
            verify=verify,
            parse=parse,
            setup_hint="Linear Settings -> API -> Webhooks",
        )
