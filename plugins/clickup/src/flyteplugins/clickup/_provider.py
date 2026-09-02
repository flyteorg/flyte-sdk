"""ClickUp webhook verification and payload normalization."""

from __future__ import annotations

from typing import Mapping

from flyte.extras.webhooks import (
    Provider,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)

#: Environment variable this provider reads its secret from by default.
DEFAULT_SECRET_ENV = "CLICKUP_WEBHOOK_SECRET"


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Clickup-Signature` HMAC over the raw body."""
    signature = lower_headers(headers).get("x-clickup-signature")
    if not signature:
        return False
    return constant_time_equals(hex_hmac_sha256(secret, body), signature.strip())


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a ClickUp delivery into a `WebhookEvent`."""
    payload = json_body(body)
    task = payload.get("task") or {}
    # ClickUp puts the list id at the top level on list-scoped events and only on
    # the nested task for task-scoped ones; read both or a `scopes` allowlist
    # cannot attribute task events.
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


class ClickUpProvider(Provider):
    """ClickUp's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.clickup import ClickUpProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[ClickUpProvider()])
    ```

    Args:
        secret_env: Environment variable holding the secret, mounted from a
            `flyte.Secret`. Override only if you store it under a non-standard
            name; the default is what the docs and examples assume.
    """

    def __init__(self, *, secret_env: str = DEFAULT_SECRET_ENV) -> None:
        super().__init__(
            name="clickup",
            secret_env=secret_env,
            verify=verify,
            parse=parse,
            setup_hint="Space Settings -> Integrations -> Webhooks",
        )
