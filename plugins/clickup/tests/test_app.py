"""Tests for the ClickUp app environment (dashboard + webhook receiver)."""

from __future__ import annotations

import pytest
import respx
from conftest import API_BASE, task_payload, webhook_body, webhook_headers
from fastapi.testclient import TestClient

from flyteplugins.clickup import ClickUpAppEnvironment


@pytest.fixture
def env():
    return ClickUpAppEnvironment(name="clickup-test-app")


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("CLICKUP_TOKEN", raising=False)
    data = client.get("/api/status").json()
    assert data["token_mounted"] is False
    monkeypatch.setenv("CLICKUP_TOKEN", "k")
    assert client.get("/api/status").json()["token_mounted"] is True


def test_dashboard_renders_instructions(client, webhook_secret):
    text = client.get("/").text
    assert "Setup instructions" in text
    assert "flyte create secret CLICKUP_TOKEN" in text
    assert "API Token" in text
    assert "/webhook" in text


def test_verify_credentials_success(client, token):
    with respx.mock(base_url=API_BASE) as router:
        router.get("/user").respond(json={"user": {"id": 1, "username": "amy", "email": "a@x"}})
        data = client.post("/api/verify").json()
    assert data == {"ok": True, "username": "amy", "email": "a@x"}


def test_verify_credentials_missing_token(client, monkeypatch):
    monkeypatch.delenv("CLICKUP_TOKEN", raising=False)
    data = client.post("/api/verify").json()
    assert data["ok"] is False


def test_rejects_bad_signature(client, webhook_secret):
    body = webhook_body(task_payload())
    response = client.post("/webhook", content=body, headers={"x-clickup-signature": "0" * 64})
    assert response.status_code == 401


def test_rejects_when_secret_missing(client, monkeypatch):
    monkeypatch.delenv("CLICKUP_WEBHOOK_SECRET", raising=False)
    body = webhook_body(task_payload())
    response = client.post("/webhook", content=body, headers=webhook_headers(body, "whatever"))
    assert response.status_code == 503


def test_dispatches_handler_and_records_event(client, env, webhook_secret):
    seen = []

    @env.on_event("taskCreated")
    async def handler(event):
        seen.append(event)
        return {"task": event.task_id}

    body = webhook_body(task_payload(event="taskCreated"))
    response = client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    data = response.json()
    assert data["ok"] is True
    assert data["event"] == "taskCreated"
    assert data["results"] == {"handler": {"task": "t1"}}
    assert len(seen) == 1

    events = client.get("/api/events").json()
    assert events[0]["task_id"] == "t1"
    assert "payload" not in events[0]


def test_list_allowlist_skips_dispatch(webhook_secret):
    env = ClickUpAppEnvironment(name="clickup-allowlist", list_ids=["l9"])
    hits = []

    @env.on_event("")
    async def handler(event):
        hits.append(event)

    test_client = TestClient(env.app)
    body = webhook_body(task_payload(list_id="l1"))
    response = test_client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
    assert hits == []


def test_allow_unsigned_events_when_configured(webhook_secret, monkeypatch):
    monkeypatch.delenv("CLICKUP_WEBHOOK_SECRET", raising=False)
    env = ClickUpAppEnvironment(name="clickup-unsigned", require_signature=False)
    test_client = TestClient(env.app)
    body = webhook_body(task_payload())
    response = test_client.post("/webhook", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 200


def test_dashboard_shows_the_most_recent_events(env, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    from flyteplugins.clickup import ClickUpEvent

    for i in range(30):
        env.recent_events.append(ClickUpEvent(event="taskCreated", task_name=f"Task {i}"))
    text = client.get("/").text
    assert "Task 29" in text
    assert "Task 0</td>" not in text


def test_allowlist_drops_events_it_cannot_attribute(webhook_secret):
    """An allowlist must not pass through an event it cannot attribute to a list."""
    from conftest import task_payload, webhook_body, webhook_headers

    allowlisted = TestClient(ClickUpAppEnvironment(name="clickup-allowlist", list_ids=["l1"]).app)

    payload = task_payload(list_id="l1")
    body = webhook_body(payload)
    assert (
        "skipped"
        not in allowlisted.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret)).json()
    )

    payload = task_payload()
    del payload["list_id"]
    payload["task"].pop("list", None)
    body = webhook_body(payload)
    response = allowlisted.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
