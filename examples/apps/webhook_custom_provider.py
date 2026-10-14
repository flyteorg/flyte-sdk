"""Support a product this family does not ship a plugin for.

A provider is small: say which environment variable holds its secret, how to
verify a delivery, and how to turn a payload into a `WebhookEvent`. Core does
the rest — the app, the dashboard, dispatch, the scope allowlist, and idempotent
launching.

Run it without an account:

    python custom_provider.py --local

That posts a signed sample delivery through the app in-process, so you see
verification, normalization, and dispatch end to end.

Once it works, move it into its own `flyteplugins-webhooks-<product>` package
beside the others and add the one-line conformance test:

    from flyte.extras.webhooks.testing import assert_provider_conforms
    import flyteplugins.webhooks.acme as plugin

    def test_conformance():
        assert_provider_conforms(plugin)
"""

import hashlib
import hmac
import json
import os
import sys
from typing import ClassVar, Mapping

import flyte
from flyte.extras.webhooks import (
    EventType,
    Provider,
    WebhookAppEnvironment,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)


class Ticket(EventType):
    """Acme's ticket events. `ANY` matches every action on the type."""

    ANY = "ticket"
    OPENED = "ticket.opened"
    CLOSED = "ticket.closed"


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify Acme's hex HMAC-SHA256 over the raw body.

    Use `constant_time_equals` rather than `hmac.compare_digest` directly: the
    latter raises `TypeError` on `str` operands containing non-ASCII, and this
    header comes off the wire, so a crafted one would turn a clean 401 into a
    500.
    """
    signature = lower_headers(headers).get("x-acme-signature")
    if not signature:
        return False
    return constant_time_equals(hex_hmac_sha256(secret, body), signature.strip())


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize an Acme delivery.

    Fill in `resource_id` and `occurred_at` wherever the product gives them:
    together they are the dedupe key, and without a timestamp every later change
    to one resource collapses onto the first one's key and never launches.
    """
    payload = json_body(body)
    ticket = payload.get("ticket") or {}
    return WebhookEvent(
        provider="acme",
        event_type="ticket",
        action=payload.get("action"),
        delivery_id=str(payload.get("delivery_id") or ""),
        resource_id=str(ticket.get("id")) if ticket.get("id") is not None else None,
        occurred_at=ticket.get("updated_at"),
        scope=ticket.get("project"),
        title=ticket.get("subject"),
        url=ticket.get("url"),
        payload=payload,
    )


class AcmeProvider(Provider):
    """Acme's webhook provider, with its defaults pre-wired.

    Users then write `providers=[AcmeProvider()]`. The app mounts
    `default_secret_env` for them; `secret_env=` is there for anyone storing the
    secret under a different name.
    """

    default_secret_env: ClassVar[str] = "ACME_WEBHOOK_SECRET"

    def __init__(self, *, secret_env: str | None = None) -> None:
        super().__init__(
            name="acme",
            secret_env=secret_env or self.default_secret_env,
            verify=verify,
            parse=parse,
            setup_hint="Acme Settings -> Webhooks",
        )


app_env = WebhookAppEnvironment(
    name="acme-webhooks",
    providers=[AcmeProvider()],
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn"),
)


@app_env.on_event(Ticket.OPENED)
async def on_ticket_opened(event):
    return {"saw": event.qualified_type, "resource": event.resource_id, "dedupe_key": event.dedupe_key()}


#: A realistic delivery, the same thing a shipped plugin exports as SAMPLE_DELIVERY.
SAMPLE_BODY = json.dumps(
    {
        "action": "opened",
        "delivery_id": "d-1",
        "ticket": {
            "id": 42,
            "subject": "Printer on fire",
            "project": "SUPPORT",
            "updated_at": "2024-01-01T00:00:00Z",
            "url": "https://acme.example/t/42",
        },
    }
).encode()


def _try_locally() -> None:
    from fastapi.testclient import TestClient

    secret = os.environ.setdefault(AcmeProvider.default_secret_env, "local-trial-secret")
    headers = {"X-Acme-Signature": hmac.new(secret.encode(), SAMPLE_BODY, hashlib.sha256).hexdigest()}
    assert app_env.app is not None  # built in __post_init__
    client = TestClient(app_env.app)

    print("POST /webhook/acme  (signed with a throwaway secret)")
    response = client.post("/webhook/acme", content=SAMPLE_BODY, headers=headers)
    print(f"  {response.status_code}  {response.json()}\n")

    print("an unsigned delivery is refused:")
    bad = client.post("/webhook/acme", content=SAMPLE_BODY, headers={})
    print(f"  {bad.status_code}  {bad.json()}")


if __name__ == "__main__":
    if "--local" in sys.argv:
        _try_locally()
    else:
        flyte.init_from_config()
        handle = flyte.serve(app_env)
        handle.activate(wait=True)
        print(f"Dashboard ready at {handle.endpoint}")
