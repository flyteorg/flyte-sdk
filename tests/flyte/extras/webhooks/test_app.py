"""Tests for the shared receiver app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent

from ._stub import STUB, UNSIGNED, Thing, body_of, stub_headers, thing_payload


@pytest.fixture
def app():
    return WebhookAppEnvironment(name="core-test", providers=[STUB, UNSIGNED])


@pytest.fixture
def client(app):
    return TestClient(app.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_a_verified_delivery_is_normalized(client, secrets):
    body = body_of(thing_payload())
    data = client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()
    assert data["provider"] == "stub"
    assert data["event"] == "thing.created"


def test_a_bad_signature_is_401(client, secrets):
    body = body_of(thing_payload())
    assert client.post("/webhook/stub", content=body, headers={"X-Stub-Signature": "deadbeef"}).status_code == 401


def test_a_missing_secret_is_503(client, monkeypatch):
    monkeypatch.delenv("STUB_WEBHOOK_SECRET", raising=False)
    body = body_of(thing_payload())
    assert client.post("/webhook/stub", content=body, headers=stub_headers(body)).status_code == 503


def test_an_unconfigured_provider_is_404(client, secrets):
    body = body_of(thing_payload())
    response = client.post("/webhook/github", content=body, headers=stub_headers(body))
    assert response.status_code == 404
    assert "configured" in response.json()["detail"]


def test_a_handshake_is_answered_before_verification(client, secrets):
    """Products send these to prove reachability, before any secret is in play."""
    response = client.post("/webhook/stub", content=b"{}", headers={"X-Stub-Event": "ping"})
    assert response.json() == {"ok": True, "ping": True}


def test_handlers_run_and_report_results(app, client, secrets):
    @app.on_event(Thing.CREATED)
    async def handler(event):
        return {"saw": event.resource_id}

    body = body_of(thing_payload())
    data = client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()
    assert data["results"]["handler"] == {"saw": "thing-1"}


def test_a_pattern_only_matches_its_own_event(app, client, secrets):
    @app.on_event(Thing.UPDATED)
    async def handler(event):  # pragma: no cover - must not run
        return {"ran": True}

    body = body_of(thing_payload(action="created"))
    assert client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()["handlers_run"] == []


def test_any_matches_every_action(app, client, secrets):
    @app.on_event(Thing.ANY)
    async def handler(event):
        return {"ok": True}

    for action in ("created", "updated"):
        body = body_of(thing_payload(action=action))
        data = client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()
        assert data["handlers_run"] == ["handler"], action


def test_a_failing_handler_is_reported_not_raised(app, client, secrets):
    @app.on_event("")
    async def boom(event):
        raise RuntimeError("handler exploded")

    body = body_of(thing_payload())
    response = client.post("/webhook/stub", content=body, headers=stub_headers(body))
    assert response.status_code == 200
    assert "handler exploded" in response.json()["errors"]["boom"]


def test_scope_allowlist_skips_other_scopes_and_unattributable_events(secrets):
    """An allowlist cannot vouch for an event it cannot attribute, so both are skipped."""
    app = WebhookAppEnvironment(name="t", providers=[STUB], scopes=["workspace-1"])
    client = TestClient(app.app)

    body = body_of(thing_payload(scope="workspace-1"))
    assert "skipped" not in client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()

    body = body_of(thing_payload(scope="somewhere-else"))
    assert (
        "not in allowlist" in client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()["skipped"]
    )

    payload = thing_payload()
    del payload["scope"]
    body = body_of(payload)
    assert (
        "not in allowlist" in client.post("/webhook/stub", content=body, headers=stub_headers(body)).json()["skipped"]
    )


def test_dashboard_lists_each_provider_and_is_honest_about_signing(client, secrets):
    text = client.get("/").text
    assert "/webhook/stub" in text
    assert "STUB_WEBHOOK_SECRET mounted" in text
    assert "provider does not sign" in text, "an unsigned provider must be labelled as such"


def test_dashboard_shows_the_most_recent_events(app, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    for i in range(30):
        app.recent_events.append(WebhookEvent(provider="stub", event_type="thing", resource_id=f"r-{i}"))
    text = client.get("/").text
    assert "r-29" in text
    assert "r-0<" not in text


def test_status_reports_each_provider(client, secrets):
    data = client.get("/api/status").json()
    assert {p["name"] for p in data["providers"]} == {"stub", "unsigned"}
    assert {p["signed"] for p in data["providers"]} == {True, False}


def test_an_app_needs_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one provider"):
        WebhookAppEnvironment(name="t", providers=[])


def test_providers_must_be_provider_instances():
    with pytest.raises(TypeError, match="must be Provider instances"):
        WebhookAppEnvironment(name="t", providers=["github"])


def test_two_providers_cannot_share_a_name():
    """Each name owns one route, so a collision would silently shadow one."""
    with pytest.raises(ValueError, match="more than one provider named"):
        WebhookAppEnvironment(name="t", providers=[STUB, STUB])
