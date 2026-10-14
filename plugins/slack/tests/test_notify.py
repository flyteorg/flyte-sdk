"""notify: the Slack Web API calls and the token-free response_url path."""

from __future__ import annotations

import json

import httpx
import pytest

from flyteplugins.slack import notify


@pytest.fixture
def slack_api(monkeypatch):
    """Capture requests and answer like Slack; yield the captured list."""
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1700000000.000200", "channel": "C1"})

    monkeypatch.setattr(notify, "_transport", httpx.MockTransport(handle))
    monkeypatch.setenv(notify.TOKEN_ENV, "xoxb-test-token")
    return captured


async def test_post_carries_the_token_and_returns_the_ts(slack_api):
    ts = await notify.post("C1", "deploy started", thread_ts="1700000000.000100")
    assert ts == "1700000000.000200"
    request = slack_api[0]
    assert str(request.url) == "https://slack.com/api/chat.postMessage"
    assert request.headers["authorization"] == "Bearer xoxb-test-token"
    body = json.loads(request.content)
    assert body == {"channel": "C1", "text": "deploy started", "thread_ts": "1700000000.000100"}


async def test_update_addresses_the_message_by_ts(slack_api):
    await notify.update("C1", "1700000000.000200", "deploy finished")
    body = json.loads(slack_api[0].content)
    assert str(slack_api[0].url).endswith("/chat.update")
    assert body["ts"] == "1700000000.000200"


async def test_slack_saying_no_raises_with_its_error_code(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    monkeypatch.setattr(notify, "_transport", httpx.MockTransport(refuse))
    monkeypatch.setenv(notify.TOKEN_ENV, "xoxb-test-token")
    with pytest.raises(notify.SlackApiError, match="channel_not_found"):
        await notify.post("C_NOPE", "hello")


async def test_a_missing_token_names_the_secret_to_mount(monkeypatch):
    monkeypatch.delenv(notify.TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="SLACK_BOT_TOKEN"):
        await notify.post("C1", "hello")


async def test_respond_needs_no_token(slack_api, monkeypatch):
    # No token anywhere: response_url is its own authorization.
    monkeypatch.delenv(notify.TOKEN_ENV, raising=False)
    await notify.respond(
        "https://hooks.slack.com/actions/T1/123/abc",
        text="*approve* — decided by <@U1>",
        replace_original=True,
    )
    request = slack_api[0]
    assert str(request.url) == "https://hooks.slack.com/actions/T1/123/abc"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"text": "*approve* — decided by <@U1>", "replace_original": True}
