"""Tests for the Slack app environment (dashboard + Events API receiver)."""

from __future__ import annotations

import pytest
import respx
from conftest import event_body, event_headers, message_event
from fastapi.testclient import TestClient

from flyteplugins.slack import SlackAppEnvironment


@pytest.fixture
def env():
    return SlackAppEnvironment(name="slack-test-app")


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    data = client.get("/api/status").json()
    assert data["bot_token_mounted"] is False
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb")
    assert client.get("/api/status").json()["bot_token_mounted"] is True


def test_dashboard_renders_instructions(client, signing_secret):
    text = client.get("/").text
    assert "Setup instructions" in text
    assert "flyte create secret SLACK_BOT_TOKEN" in text
    assert "url_verification" in text
    assert "/events" in text


def test_verify_credentials_success(client, bot_token):
    with respx.mock(base_url="https://slack.com/api") as router:
        router.post("/auth.test").respond(json={"ok": True, "user": "flytebot", "team": "acme"})
        data = client.post("/api/verify").json()
    assert data["ok"] is True
    assert data["user"] == "flytebot"


def test_verify_credentials_missing_token(client, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    data = client.post("/api/verify").json()
    assert data["ok"] is False


def test_url_verification_challenge(client):
    response = client.post("/events", content=b'{"type": "url_verification", "challenge": "xyz"}')
    assert response.status_code == 200
    assert response.json() == {"challenge": "xyz"}


def test_rejects_bad_signature(client, signing_secret):
    body = event_body(message_event())
    headers = event_headers(body, secret="wrong-secret")
    assert client.post("/events", content=body, headers=headers).status_code == 401


def test_rejects_when_secret_missing(client, monkeypatch):
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    body = event_body(message_event())
    response = client.post("/events", content=body, headers=event_headers(body))
    assert response.status_code == 503


def test_dispatches_handler_and_records_event(client, env, signing_secret):
    seen = []

    @env.on_event("message")
    async def handler(event):
        seen.append(event)
        return {"channel": event.channel}

    body = event_body(message_event(channel="C777"))
    response = client.post("/events", content=body, headers=event_headers(body))
    data = response.json()
    assert data["ok"] is True
    assert data["event"] == "message"
    assert data["results"] == {"handler": {"channel": "C777"}}
    assert len(seen) == 1

    events = client.get("/api/events").json()
    assert events[0]["channel"] == "C777"
    assert "payload" not in events[0]


def test_channel_allowlist_skips_dispatch(signing_secret):
    env = SlackAppEnvironment(name="slack-allowlist", channels=["C1"])
    hits = []

    @env.on_event("")
    async def handler(event):
        hits.append(event)

    test_client = TestClient(env.app)
    body = event_body(message_event(channel="C999"))
    response = test_client.post("/events", content=body, headers=event_headers(body))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
    assert hits == []


def test_allow_unsigned_events_when_configured(monkeypatch):
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    env = SlackAppEnvironment(name="slack-unsigned", require_signature=False)
    test_client = TestClient(env.app)
    body = event_body(message_event())
    response = test_client.post("/events", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 200
