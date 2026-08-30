"""Tests for the Slack Web API client."""

from __future__ import annotations

import httpx
import pytest

from flyteplugins.slack import MissingCredentialsError, SlackAPIError, SlackClient


async def test_post_message_returns_ts_and_permalink(slack_api):
    slack_api.post("/chat.postMessage").respond(json={"ok": True, "channel": "C123", "ts": "111.222"})
    slack_api.get("/chat.getPermalink").respond(json={"ok": True, "permalink": "https://slack/archives/C123/p111"})

    async with SlackClient(bot_token="xoxb") as client:
        result = await client.post_message("C123", "hello")
    assert result["ts"] == "111.222"
    assert result["permalink"].endswith("p111")


async def test_post_message_ok_without_permalink(slack_api):
    slack_api.post("/chat.postMessage").respond(json={"ok": True, "channel": "C123", "ts": "111.222"})
    slack_api.get("/chat.getPermalink").respond(json={"ok": False, "error": "message_not_found"})

    async with SlackClient(bot_token="xoxb") as client:
        result = await client.post_message("C123", "hello")
    assert result == {"channel": "C123", "ts": "111.222"}


async def test_thread_reply_passes_thread_ts(slack_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"ok": True, "channel": "C123", "ts": "2.0"})

    slack_api.post("/chat.postMessage").mock(side_effect=capture)
    slack_api.get("/chat.getPermalink").respond(json={"ok": True, "permalink": "p"})

    async with SlackClient(bot_token="xoxb") as client:
        await client.reply_in_thread("C123", "1.0", "reply")
    assert captured["body"]["thread_ts"] == "1.0"


async def test_api_error_on_ok_false(slack_api):
    slack_api.post("/chat.postMessage").respond(json={"ok": False, "error": "channel_not_found"})
    async with SlackClient(bot_token="xoxb") as client:
        with pytest.raises(SlackAPIError) as excinfo:
            await client.post_message("Cnope", "hi")
    assert excinfo.value.error == "channel_not_found"


async def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(MissingCredentialsError) as excinfo:
        async with SlackClient():
            pass
    assert "SLACK_BOT_TOKEN" in str(excinfo.value)


async def test_retries_on_429(slack_api):
    route = slack_api.get("/users.info")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"ok": True, "user": {"id": "U1", "name": "amy"}}),
    ]
    from flyteplugins.slack import Config

    async with SlackClient(Config(retry_backoff=0.0), bot_token="xoxb") as client:
        user = await client.get_user("U1")
    assert user["name"] == "amy"
    assert route.call_count == 2


async def test_get_channel_history_newest_last(slack_api):
    slack_api.get("/conversations.history").respond(
        json={"ok": True, "messages": [{"ts": "2", "text": "newer"}, {"ts": "1", "text": "older"}]}
    )
    async with SlackClient(bot_token="xoxb") as client:
        history = await client.get_channel_history("C123")
    assert [m["text"] for m in history] == ["older", "newer"]


async def test_get_thread(slack_api):
    slack_api.get("/conversations.replies").respond(
        json={
            "ok": True,
            "messages": [{"ts": "1.0", "text": "root"}, {"ts": "1.1", "thread_ts": "1.0", "text": "reply"}],
        }
    )
    async with SlackClient(bot_token="xoxb") as client:
        thread = await client.get_thread("C123", "1.0")
    assert len(thread) == 2
    assert thread[1]["thread_ts"] == "1.0"


async def test_add_reaction_strips_colons(slack_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"ok": True})

    slack_api.post("/reactions.add").mock(side_effect=capture)
    async with SlackClient(bot_token="xoxb") as client:
        assert await client.add_reaction("C123", "1.0", ":bug:") is True
    assert captured["body"]["name"] == "bug"


async def test_list_channels_simplified(slack_api):
    slack_api.get("/conversations.list").respond(
        json={"ok": True, "channels": [{"id": "C1", "name": "general", "is_private": False}]}
    )
    async with SlackClient(bot_token="xoxb") as client:
        channels = await client.list_channels()
    assert channels == [{"id": "C1", "name": "general", "is_private": False, "is_member": False}]


async def test_create_channel(slack_api):
    slack_api.post("/conversations.create").respond(json={"ok": True, "channel": {"id": "C99", "name": "flyte-alerts"}})
    async with SlackClient(bot_token="xoxb") as client:
        channel = await client.create_channel("flyte-alerts")
    assert channel == {"id": "C99", "name": "flyte-alerts"}


async def test_client_requires_context_manager():
    client = SlackClient(bot_token="xoxb")
    with pytest.raises(RuntimeError):
        await client.get_user("U1")
