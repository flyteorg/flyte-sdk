"""Tests for the Jira app environment (dashboard + webhook receiver)."""

from __future__ import annotations

import pytest
import respx
from conftest import API_BASE, json_body, webhook_payload
from fastapi.testclient import TestClient

from flyteplugins.jira import JiraAppEnvironment


@pytest.fixture
def env():
    return JiraAppEnvironment(name="jira-test-app")


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
    data = client.get("/api/status").json()
    assert data["api_token_mounted"] is False
    monkeypatch.setenv("JIRA_API_TOKEN", "t")
    assert client.get("/api/status").json()["api_token_mounted"] is True


def test_dashboard_renders_instructions(client, webhook_token):
    text = client.get("/").text
    assert "Setup instructions" in text
    assert "flyte create secret JIRA_API_TOKEN" in text
    assert "id.atlassian.net" in text
    assert "not signed" in text
    assert "/webhook" in text


def test_verify_credentials_success(client, creds):
    with respx.mock(base_url=API_BASE) as router:
        router.get("/myself").respond(json={"displayName": "Bot", "emailAddress": "bot@acme.com"})
        data = client.post("/api/verify").json()
    assert data == {"ok": True, "display_name": "Bot", "email": "bot@acme.com"}


def test_verify_credentials_missing(client, monkeypatch):
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
    data = client.post("/api/verify").json()
    assert data["ok"] is False
    assert "JIRA_API_TOKEN" in data["error"]


def test_rejects_missing_webhook_token(client, webhook_token):
    body = json_body(webhook_payload())
    response = client.post("/webhook", content=body)
    assert response.status_code == 401


def test_rejects_bad_webhook_token(client, webhook_token):
    body = json_body(webhook_payload())
    response = client.post("/webhook", content=body, headers={"X-Webhook-Token": "wrong"})
    assert response.status_code == 401


def test_rejects_when_secret_not_mounted(client, monkeypatch):
    monkeypatch.delenv("JIRA_WEBHOOK_TOKEN", raising=False)
    body = json_body(webhook_payload())
    response = client.post("/webhook", content=body, headers={"X-Webhook-Token": "anything"})
    assert response.status_code == 503


def test_dispatches_handler_and_records_event(client, env, webhook_token):
    seen = []

    @env.on_event("jira:issue_created")
    async def handler(event):
        seen.append(event)
        return {"issue": event.issue_key}

    body = json_body(webhook_payload(event="jira:issue_created", key="PROJ-77"))
    response = client.post("/webhook", content=body, headers={"X-Webhook-Token": webhook_token})
    data = response.json()
    assert data["ok"] is True
    assert data["event"] == "jira:issue_created"
    assert data["results"] == {"handler": {"issue": "PROJ-77"}}
    assert len(seen) == 1

    events = client.get("/api/events").json()
    assert events[0]["issue_key"] == "PROJ-77"
    assert "payload" not in events[0]


def test_project_allowlist_skips_dispatch(webhook_token):
    env = JiraAppEnvironment(name="jira-allowlist", project_keys=["OTHER"])
    hits = []

    @env.on_event("")
    async def handler(event):
        hits.append(event)

    test_client = TestClient(env.app)
    body = json_body(webhook_payload())  # project PROJ
    response = test_client.post("/webhook", content=body, headers={"X-Webhook-Token": webhook_token})
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
    assert hits == []


def test_allow_tokenless_events_when_configured(webhook_token, monkeypatch):
    monkeypatch.delenv("JIRA_WEBHOOK_TOKEN", raising=False)
    env = JiraAppEnvironment(name="jira-open", require_webhook_token=False)
    test_client = TestClient(env.app)
    body = json_body(webhook_payload())
    response = test_client.post("/webhook", content=body)
    assert response.status_code == 200


def test_dashboard_shows_the_most_recent_events(env, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    from flyteplugins.jira import JiraEvent

    for i in range(30):
        env.recent_events.append(JiraEvent(webhook_event="jira:issue_created", issue_key=f"ENG-{i}"))
    text = client.get("/").text
    assert "ENG-29" in text
    assert "ENG-0<" not in text


def test_allowlist_drops_events_it_cannot_attribute(webhook_token):
    """An allowlist must not pass through an event it cannot attribute to a project."""
    from conftest import json_body, webhook_payload

    allowlisted = TestClient(JiraAppEnvironment(name="jira-allowlist", project_keys=["PROJ"]).app)
    headers = {"X-Webhook-Token": webhook_token}

    body = json_body(webhook_payload())
    assert "skipped" not in allowlisted.post("/webhook", content=body, headers=headers).json()

    payload = webhook_payload()
    payload["issue"]["fields"].pop("project", None)
    body = json_body(payload)
    response = allowlisted.post("/webhook", content=body, headers=headers)
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
