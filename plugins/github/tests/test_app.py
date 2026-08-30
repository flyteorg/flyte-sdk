"""Tests for the GitHub app environment (dashboard + webhook receiver)."""

from __future__ import annotations

import pytest
import respx
from conftest import pr_payload, webhook_body, webhook_headers
from fastapi.testclient import TestClient

from flyteplugins.github import GitHubAppEnvironment


@pytest.fixture
def env():
    return GitHubAppEnvironment(name="gh-test-app")


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    data = client.get("/api/status").json()
    assert data["token_mounted"] is False
    assert data["webhook_secret_mounted"] is False

    monkeypatch.setenv("GITHUB_TOKEN", "t")
    data = client.get("/api/status").json()
    assert data["token_mounted"] is True


def test_dashboard_renders_instructions(client, webhook_secret):
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "Setup instructions" in text
    assert "flyte create secret GITHUB_TOKEN" in text
    assert "/webhook" in text
    assert "GITHUB_WEBHOOK_SECRET mounted" in text


def test_verify_credentials_success(client, token):
    with respx.mock(base_url="https://api.github.com") as router:
        router.get("/user").respond(json={"login": "octocat"}, headers={"x-oauth-scopes": "repo, read:org"})
        data = client.post("/api/verify").json()
    assert data == {"ok": True, "login": "octocat", "scopes": "repo, read:org"}


def test_verify_credentials_missing_token(client, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    data = client.post("/api/verify").json()
    assert data["ok"] is False
    assert "GITHUB_TOKEN" in data["error"]


def test_webhook_ping(client, webhook_secret):
    response = client.post(
        "/webhook",
        content=b'{"zen": "design for failure"}',
        headers=webhook_headers(b'{"zen": "design for failure"}', webhook_secret, event="ping"),
    )
    assert response.status_code == 200
    assert response.json()["ping"] is True


def test_webhook_rejects_bad_signature(client, webhook_secret):
    body = webhook_body(pr_payload())
    headers = webhook_headers(body, webhook_secret)
    headers["X-Hub-Signature-256"] = "sha256=" + "0" * 64
    response = client.post("/webhook", content=body, headers=headers)
    assert response.status_code == 401


def test_webhook_rejects_when_secret_missing(client, monkeypatch):
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    body = webhook_body(pr_payload())
    response = client.post("/webhook", content=body, headers=webhook_headers(body, "whatever"))
    assert response.status_code == 503


def test_webhook_dispatches_handler_and_records_event(client, env, webhook_secret):
    seen = []

    @env.on_event("pull_request.opened")
    async def handler(event):
        seen.append(event)
        return {"handled": event.number}

    body = webhook_body(pr_payload(number=7))
    response = client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["event"] == "pull_request.opened"
    assert data["results"] == {"handler": {"handled": 7}}
    assert len(seen) == 1

    events = client.get("/api/events").json()
    assert events[0]["number"] == 7
    assert "payload" not in events[0]


def test_webhook_handler_pattern_matching(client, env, webhook_secret):
    hits = []

    @env.on_event("issues")
    async def issues_handler(event):
        hits.append("issues")

    @env.on_event("")
    async def all_handler(event):
        hits.append("all")

    body = webhook_body({"action": "opened", "repository": {"full_name": "o/r"}})
    client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret, event="push"))
    # push does not match "issues", but matches the catch-all
    assert hits == ["all"]


def test_webhook_repo_allowlist_skips_dispatch(client, webhook_secret):
    env = GitHubAppEnvironment(name="gh-allowlist", repos=["other/repo"])
    hits = []

    @env.on_event("")
    async def handler(event):
        hits.append(event)

    test_client = TestClient(env.app)
    body = webhook_body(pr_payload(repo="octo/repo"))
    response = test_client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    assert "not in allowlist" in response.json()["skipped"]
    assert hits == []


def test_webhook_handler_error_reported(client, env, webhook_secret):
    @env.on_event("pull_request")
    async def broken(event):
        raise RuntimeError("boom")

    body = webhook_body(pr_payload())
    response = client.post("/webhook", content=body, headers=webhook_headers(body, webhook_secret))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "boom" in data["errors"]["broken"]


def test_allow_unsigned_events_when_configured(webhook_secret, monkeypatch):
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    env = GitHubAppEnvironment(name="gh-unsigned", require_signature=False)
    test_client = TestClient(env.app)
    body = webhook_body(pr_payload())
    response = test_client.post(
        "/webhook",
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "1"},
    )
    assert response.status_code == 200
