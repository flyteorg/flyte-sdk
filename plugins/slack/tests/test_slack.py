"""Slack-specific verification and normalization, beyond conformance."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from flyteplugins.slack import SlackProvider, events, parse, verify

SECRET = "slack-secret"


def _headers(body: bytes, timestamp: int | None = None) -> dict:
    ts = str(timestamp if timestamp is not None else int(time.time()))
    base = b"v0:" + ts.encode() + b":" + body
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": "v0=" + hmac.new(SECRET.encode(), base, hashlib.sha256).hexdigest(),
    }


def test_a_stale_timestamp_is_rejected_as_a_replay():
    body = b"{}"
    assert verify(body, _headers(body), SECRET) is True
    assert verify(body, _headers(body, timestamp=0), SECRET) is False


def test_the_signature_covers_the_raw_bytes():
    """Decoding the body and re-encoding it would corrupt bytes Slack signed."""
    body = b'{"text": "\xff\xfe"}'
    assert verify(body, _headers(body), SECRET) is True


def test_a_message_subtype_becomes_the_qualified_type():
    body = json.dumps(
        {"event_id": "E1", "event": {"type": "message", "subtype": "message_changed", "channel": "C1", "ts": "1.0"}}
    ).encode()
    assert parse(_headers(body), body).qualified_type == events.Message.CHANGED


def test_messages_are_keyed_per_message_not_per_thread():
    """One run per message by default; pass your own key to collapse a thread."""

    def message(ts: str):
        body = json.dumps(
            {"event_id": "E", "event": {"type": "message", "channel": "C1", "ts": ts, "thread_ts": "1.0"}}
        ).encode()
        return parse(_headers(body), body)

    assert message("2.0").dedupe_key() != message("3.0").dedupe_key()


def test_the_url_verification_handshake_is_echoed():
    body = json.dumps({"type": "url_verification", "challenge": "abc123"}).encode()
    assert SlackProvider().handshake({}, body) == {"challenge": "abc123"}
    assert SlackProvider().handshake({}, b'{"type": "event_callback"}') is None
