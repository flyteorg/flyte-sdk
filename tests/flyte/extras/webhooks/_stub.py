"""A stub provider, so the receiver is tested without any product package.

Core is tested against a stub provider rather than a real one, so it has no
dependency on any `flyteplugins-webhooks-<product>` package. The real providers
prove themselves through `assert_provider_conforms` in their own suites.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping

from flyte.extras.webhooks import (
    EventType,
    Provider,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)

STUB_SECRET = "stub-secret"


class Thing(EventType):
    """Event constants for the stub provider."""

    ANY = "thing"
    CREATED = "thing.created"
    UPDATED = "thing.updated"


def stub_verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    signature = lower_headers(headers).get("x-stub-signature")
    return bool(signature) and constant_time_equals(hex_hmac_sha256(secret, body), signature.strip())


def stub_handshake(headers: Mapping[str, str], body: bytes) -> dict[str, Any] | None:
    if lower_headers(headers).get("x-stub-event") == "ping":
        return {"ok": True, "ping": True}
    return None


def stub_parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    payload = json_body(body)
    return WebhookEvent(
        provider="stub",
        event_type=payload.get("type", "thing"),
        action=payload.get("action"),
        delivery_id=payload.get("delivery_id", ""),
        resource_id=payload.get("id"),
        occurred_at=payload.get("updated_at"),
        scope=payload.get("scope"),
        title=payload.get("title"),
        payload=payload,
    )


STUB = Provider(
    name="stub",
    secret_env="STUB_WEBHOOK_SECRET",
    verify=stub_verify,
    parse=stub_parse,
    handshake=stub_handshake,
    setup_hint="nowhere; this provider is a test double",
)

UNSIGNED = Provider(
    name="unsigned",
    secret_env="UNSIGNED_WEBHOOK_TOKEN",
    verify=lambda body, headers, secret: constant_time_equals(
        lower_headers(headers).get("x-unsigned-token", ""), secret
    ),
    parse=lambda headers, body: WebhookEvent(provider="unsigned", event_type="ping"),
    signed=False,
)


def body_of(payload: dict) -> bytes:
    return json.dumps(payload).encode()


def stub_headers(body: bytes, secret: str = STUB_SECRET) -> dict[str, str]:
    return {"X-Stub-Signature": hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()}


def thing_payload(action: str = "created", scope: str = "workspace-1", **extra) -> dict:
    return {
        "type": "thing",
        "action": action,
        "id": "thing-1",
        "scope": scope,
        "title": "A thing",
        "delivery_id": "d-1",
        **extra,
    }
