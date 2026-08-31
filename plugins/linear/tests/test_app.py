"""Tests for the Linear app environment (dashboard + webhook receiver)."""

from __future__ import annotations

import pytest
import respx
from conftest import GRAPHQL_URL, issue_payload, webhook_body, webhook_headers
from fastapi.testclient import TestClient

from flyteplugins.linear import LinearAppEnvironment


@pytest.fixture
def env():
    return LinearAppEnvironment(name="linear-test-app")


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    data = client.get("/api/status").json()
    assert data["api_key_mounted"] is False
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    assert client.get("/api/status").json()["api_key_mounted"] is True


def test_dashboard_renders_instructions(client, webhook_secret):
    text = client.get("/").text
    assert "Setup instructions" in text
    assert "flyte create secret LINEAR_API_KEY" in text
    assert "Personal API keys" in text
    assert "/webhook" in text


def test_verify_credentials_success(client, api_key):
    with respx.mock() as router:
        router.post(GRAPHQL_URL).respond(json={"data": {"viewer": {"id": "u1", "name": "amy", "displayName": "Amy"}}})
        data = client.post("/api/verify").json()
    assert data == {"ok": True, "user": "Amy"}


def test_verify_credentials_missing_key(client, monkeypatch):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    data = client.post("/api/verify").json()
    assert data["ok"] is False


def test_rejects_bad_signature(client, webhook_secret):
    body = webhook_body(issue_payload())
    response = client.post("/webhook", content=body, headers={"X-Linear-Signature": "0" * 64})
    assert response.status_code == 401


def test_rejects_when_secret_missing(client, monkeypatch):
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    body = webhook_body(issue_payload())
    response = client.post("/webhook", content=body, headers=webhook_headers(body, "whatever"))
    assert response.status_code == 503


def test_dispatches_handler_and_records_event(client, env, webhook_secret):
    seen = []

    @env.on_event("Issue.create")
    async def handler(event):
        seen.append(event)
        return {"issue": event.entity_id}

    body = webhook_body(issue_payload(action="create"))
    response = client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    data = response.json()
    assert data["ok"] is True
    assert data["event"] == "Issue.create"
    assert data["results"] == {"handler": {"issue": "issue-uuid"}}
    assert len(seen) == 1

    events = client.get("/api/events").json()
    assert events[0]["entity_id"] == "issue-uuid"
    assert "payload" not in events[0]


def test_team_allowlist_skips_dispatch(webhook_secret):
    env = LinearAppEnvironment(name="linear-allowlist", team_ids=["team-9"])
    hits = []

    @env.on_event("")
    async def handler(event):
        hits.append(event)

    test_client = TestClient(env.app)
    body = webhook_body(issue_payload(team_id="team-1"))
    response = test_client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
    assert hits == []


def test_allow_unsigned_events_when_configured(webhook_secret, monkeypatch):
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    env = LinearAppEnvironment(name="linear-unsigned", require_signature=False)
    test_client = TestClient(env.app)
    body = webhook_body(issue_payload())
    response = test_client.post("/webhook", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 200


def test_dashboard_shows_the_most_recent_events(env, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    from flyteplugins.linear import LinearEvent

    for i in range(30):
        env.recent_events.append(LinearEvent(action="create", entity_type="Issue", title=f"Issue {i}"))
    text = client.get("/").text
    assert "Issue 29" in text
    assert "Issue 0</td>" not in text


def test_allowlist_drops_events_it_cannot_attribute(webhook_secret):
    """An allowlist must not pass through an event it cannot attribute to a team."""
    from conftest import issue_payload, webhook_body, webhook_headers

    allowlisted = TestClient(LinearAppEnvironment(name="linear-allowlist", team_ids=["team-1"]).app)

    payload = issue_payload(team_id="team-1")
    body = webhook_body(payload)
    assert (
        "skipped"
        not in allowlisted.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret)).json()
    )

    payload = issue_payload()
    del payload["data"]["teamId"]
    body = webhook_body(payload)
    response = allowlisted.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
