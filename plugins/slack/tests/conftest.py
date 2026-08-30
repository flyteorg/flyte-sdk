"""Shared fixtures for Slack plugin tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
import respx

SIGNING_SECRET = "test-signing-secret"


@pytest.fixture
def slack_api():
    """A respx router mocking https://slack.com/api."""
    with respx.mock(base_url="https://slack.com/api", assert_all_called=False) as router:
        yield router


@pytest.fixture
def bot_token(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    return "xoxb-test-token"


@pytest.fixture
def signing_secret(monkeypatch):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    return SIGNING_SECRET


def sign(body: bytes, secret: str, timestamp: int) -> str:
    """Compute the X-Slack-Signature header value for a payload."""
    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    return "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()


def message_event(channel: str = "C123", ts: str = "123.456", user: str = "U42", text: str = "hello") -> dict:
    return {
        "type": "event_callback",
        "event_id": "Ev123",
        "team_id": "T1",
        "event": {
            "type": "message",
            "channel": channel,
            "ts": ts,
            "user": user,
            "text": text,
        },
    }


def event_headers(body: bytes, secret: str = SIGNING_SECRET, timestamp: int | None = None) -> dict:
    ts = timestamp if timestamp is not None else int(time.time())
    return {
        "X-Slack-Request-Timestamp": str(ts),
        "X-Slack-Signature": sign(body, secret, ts),
        "Content-Type": "application/json",
    }


def event_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
