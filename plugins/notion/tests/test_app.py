"""Tests for the Notion app environment (dashboard + poll endpoint)."""

from __future__ import annotations

import pytest
import respx
from conftest import API_BASE, page_json, query_response
from fastapi.testclient import TestClient

from flyteplugins.notion import NotionAppEnvironment


@pytest.fixture
def env():
    return NotionAppEnvironment(name="notion-test-app", databases=["db1"])


@pytest.fixture
def client(env):
    return TestClient(env.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "healthy"}


def test_status_reports_mounted_state(client, monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    data = client.get("/api/status").json()
    assert data["token_mounted"] is False
    monkeypatch.setenv("NOTION_TOKEN", "k")
    data = client.get("/api/status").json()
    assert data["token_mounted"] is True
    assert data["databases"] == ["db1"]


def test_dashboard_renders_instructions(client, poll_token):
    text = client.get("/").text
    assert "Setup instructions" in text
    assert "flyte create secret NOTION_TOKEN" in text
    assert "Notion has no webhooks" in text or "polling" in text.lower()
    assert "/api/poll" in text


def test_verify_credentials_success(client, token):
    with respx.mock(base_url=API_BASE) as router:
        router.get("/users/me").respond(json={"id": "b", "name": "Flyte Bot", "type": "bot"})
        data = client.post("/api/verify").json()
    assert data == {"ok": True, "name": "Flyte Bot", "type": "bot"}


def test_poll_rejects_missing_poll_token(client, token, monkeypatch):
    monkeypatch.delenv("NOTION_POLL_TOKEN", raising=False)
    response = client.get("/api/poll")
    assert response.status_code == 503


def test_poll_rejects_bad_poll_token(client, token, poll_token):
    response = client.get("/api/poll", headers={"X-Poll-Token": "wrong"})
    assert response.status_code == 401


def test_poll_rejects_unconfigured_database(client, token, poll_token):
    response = client.get("/api/poll?database_id=other-db", headers={"X-Poll-Token": poll_token})
    assert response.status_code == 403


def test_poll_dispatches_handlers(client, env, token, poll_token):
    seen = []

    @env.on_event("page.edited")
    async def handler(event):
        seen.append(event)
        return {"page": event.page_id}

    with respx.mock(base_url=API_BASE) as router:
        router.post("/databases/db1/query").respond(
            json=query_response([page_json(page_id="p77", title="Updated row")])
        )
        response = client.get("/api/poll", headers={"X-Poll-Token": poll_token})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["count"] == 1
    assert data["events"][0]["page_id"] == "p77"
    assert data["results"] == {"handler:p77": {"page": "p77"}}
    assert len(seen) == 1
    assert seen[0].title == "Updated row"

    events = client.get("/api/events").json()
    assert events[0]["page_id"] == "p77"
    assert "payload" not in events[0]


def test_poll_since_parameter_forwarded(client, token, poll_token):
    captured = {}

    def capture(request):
        import json as _json

        captured["body"] = _json.loads(request.content)
        return respx.MockResponse(json=query_response([]), status_code=200)

    with respx.mock(base_url=API_BASE) as router:
        router.post("/databases/db1/query").mock(side_effect=capture)
        response = client.get("/api/poll?since=2024-05-01T00:00:00.000Z", headers={"X-Poll-Token": poll_token})
    assert response.status_code == 200
    assert captured["body"]["filter"]["last_edited_time"]["after"] == "2024-05-01T00:00:00.000Z"


def test_poll_without_token_requirement(token, monkeypatch):
    monkeypatch.delenv("NOTION_POLL_TOKEN", raising=False)
    env = NotionAppEnvironment(name="notion-open", databases=["db1"], require_poll_token=False)
    test_client = TestClient(env.app)
    with respx.mock(base_url=API_BASE) as router:
        router.post("/databases/db1/query").respond(json=query_response([]))
        response = test_client.get("/api/poll")
    assert response.status_code == 200
    assert response.json()["count"] == 0


async def test_non_ascii_poll_token_is_rejected_not_raised(env, poll_token):
    """An attacker-controlled header must yield 401, not a 500 from compare_digest.

    ASGI servers hand raw header bytes to Starlette, which decodes them as
    latin-1 — so a non-ASCII token really can reach the comparison, even though
    the httpx test client refuses to send one.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await env._handle_poll(None, None, "\xff\xfe")
    assert exc.value.status_code == 401


def test_dashboard_shows_the_most_recent_events(env, client):
    """The buffer appends on the right, so the dashboard must read from the end."""
    from flyteplugins.notion import NotionEvent

    for i in range(30):
        env.recent_events.append(NotionEvent(page_id=f"page-{i}", title=f"Page {i}"))
    text = client.get("/").text
    assert "Page 29" in text
    assert "Page 0</td>" not in text
