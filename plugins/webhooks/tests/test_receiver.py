"""Tests for the unified webhook receiver."""

from __future__ import annotations

import pytest
from conftest import ALL_PROVIDERS, body_of, github_headers, pr_payload, slack_headers, slack_payload
from fastapi.testclient import TestClient

from flyteplugins.webhooks import WebhookAppEnvironment

ALL = ["github", "slack", "linear", "clickup", "jira"]


@pytest.fixture
def app():
    return WebhookAppEnvironment(name="webhooks-test", providers=ALL)


@pytest.fixture
def client(app):
    return TestClient(app.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_unknown_provider_is_404(client, secrets):
    body = body_of(pr_payload())
    assert client.post("/webhook/notion", content=body, headers=github_headers(body)).status_code == 404


def test_unconfigured_provider_is_404(secrets):
    """A provider the app was not built with must not be reachable."""
    only_github = TestClient(WebhookAppEnvironment(name="t", providers=["github"]).app)
    body = body_of(slack_payload())
    assert only_github.post("/webhook/slack", content=body, headers=slack_headers(body)).status_code == 404


@pytest.mark.parametrize(("provider", "headers_for", "payload_for", "expected"), ALL_PROVIDERS)
def test_every_provider_verifies_and_normalizes(client, secrets, provider, headers_for, payload_for, expected):
    body = body_of(payload_for())
    response = client.post(f"/webhook/{provider}", content=body, headers=headers_for(body))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["provider"] == provider
    assert data["event"] == expected


@pytest.mark.parametrize(("provider", "headers_for", "payload_for", "expected"), ALL_PROVIDERS)
def test_every_provider_rejects_a_bad_signature(client, secrets, provider, headers_for, payload_for, expected):
    body = body_of(payload_for())
    headers = headers_for(body)
    # Corrupt whichever header carries the credential.
    for key in list(headers):
        if key.lower() in (
            "x-hub-signature-256",
            "x-slack-signature",
            "x-linear-signature",
            "x-clickup-signature",
            "x-webhook-token",
        ):
            headers[key] = "sha256=deadbeef" if "hub" in key.lower() else "deadbeef"
    assert client.post(f"/webhook/{provider}", content=body, headers=headers).status_code == 401


@pytest.mark.parametrize(("provider", "headers_for", "payload_for", "expected"), ALL_PROVIDERS)
def test_every_provider_503s_when_its_secret_is_missing(
    client, monkeypatch, provider, headers_for, payload_for, expected
):
    for var in (
        "GITHUB_WEBHOOK_SECRET",
        "SLACK_SIGNING_SECRET",
        "LINEAR_WEBHOOK_SECRET",
        "CLICKUP_WEBHOOK_SECRET",
        "JIRA_WEBHOOK_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    body = body_of(payload_for())
    assert client.post(f"/webhook/{provider}", content=body, headers=headers_for(body)).status_code == 503


def test_github_ping_is_answered_without_a_signature(client, secrets):
    body = b'{"zen": "design for failure"}'
    response = client.post("/webhook/github", content=body, headers={"X-GitHub-Event": "ping"})
    assert response.json() == {"ok": True, "ping": True}


def test_slack_url_verification_is_echoed(client, secrets):
    body = body_of({"type": "url_verification", "challenge": "abc123"})
    response = client.post("/webhook/slack", content=body, headers={})
    assert response.json() == {"challenge": "abc123"}


def test_stale_slack_timestamp_is_rejected_as_a_replay(client, secrets):
    body = body_of(slack_payload())
    assert client.post("/webhook/slack", content=body, headers=slack_headers(body, timestamp=0)).status_code == 401


def test_handlers_run_and_report_results(app, client, secrets):
    from flyteplugins.webhooks import events

    @app.on_event(events.github.PullRequest.OPENED)
    async def handler(event):
        return {"saw": event.resource_id}

    body = body_of(pr_payload(number=7))
    data = client.post("/webhook/github", content=body, headers=github_headers(body)).json()
    assert data["handlers_run"] == ["handler"]
    assert data["results"]["handler"] == {"saw": "octo/repo#7"}


def test_a_failing_handler_is_reported_not_raised(app, client, secrets):
    @app.on_event("")
    async def boom(event):
        raise RuntimeError("handler exploded")

    body = body_of(pr_payload())
    response = client.post("/webhook/github", content=body, headers=github_headers(body))
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "handler exploded" in data["errors"]["boom"]


def test_scope_allowlist_skips_other_scopes_and_unattributable_events(secrets):
    """An allowlist cannot vouch for an event it cannot attribute, so both are skipped."""
    app = WebhookAppEnvironment(name="t", providers=["github"], scopes=["octo/repo"])
    client = TestClient(app.app)

    body = body_of(pr_payload(repo="octo/repo"))
    assert "skipped" not in client.post("/webhook/github", content=body, headers=github_headers(body)).json()

    body = body_of(pr_payload(repo="someone/else"))
    assert (
        "not in allowlist"
        in client.post("/webhook/github", content=body, headers=github_headers(body)).json()["skipped"]
    )

    payload = pr_payload()
    del payload["repository"]
    body = body_of(payload)
    assert (
        "not in allowlist"
        in client.post("/webhook/github", content=body, headers=github_headers(body)).json()["skipped"]
    )


def test_dashboard_lists_every_configured_provider(client, secrets):
    text = client.get("/").text
    for provider in ALL:
        assert f"/webhook/{provider}" in text
    assert "GITHUB_WEBHOOK_SECRET mounted" in text
    assert "provider does not sign" in text, "the dashboard must be honest that Jira is unsigned"


def test_dashboard_shows_the_most_recent_events(app, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    from flyteplugins.webhooks import WebhookEvent

    for i in range(30):
        app.recent_events.append(WebhookEvent(provider="github", event_type="issues", resource_id=f"r-{i}"))
    text = client.get("/").text
    assert "r-29" in text
    assert "r-0<" not in text


def test_status_reports_each_provider(client, secrets):
    data = client.get("/api/status").json()
    assert {p["name"] for p in data["providers"]} == set(ALL)
    assert all(p["secret_mounted"] for p in data["providers"])


def test_configuring_an_unknown_provider_fails_loudly():
    with pytest.raises(ValueError, match="unknown provider"):
        WebhookAppEnvironment(name="t", providers=["notion"])
