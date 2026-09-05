"""Slack-specific verification and normalization, beyond conformance."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

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


def _interaction_body(payload: dict) -> bytes:
    """Interactivity deliveries are form-encoded, the JSON under `payload`."""
    return urlencode({"payload": json.dumps(payload)}).encode()


def _block_action(action_id: str, action_ts: str = "1700000001.000000") -> dict:
    return {
        "type": "block_actions",
        "trigger_id": "13345224609.738474920.8088930838d88f008e0",
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "container": {"type": "message", "channel_id": "C1", "message_ts": "1700000000.000100"},
        "actions": [{"action_id": action_id, "action_ts": action_ts}],
        "message": {"ts": "1700000000.000100", "text": "proposed reply"},
        "response_url": "https://hooks.slack.com/actions/T1/123/abc",
    }


def test_a_block_action_registers_as_its_action_id():
    body = _interaction_body(_block_action("send_to_customer"))
    event = parse(_headers(body), body)
    assert event.qualified_type == "block_actions.send_to_customer"
    assert event.event_type == events.Interaction.BLOCK_ACTIONS
    assert event.scope == "C1"
    assert event.resource_id == "C1:1700000000.000100"
    assert event.actor == "U1"
    # Handlers get Slack's full JSON — actions, container, message, response_url.
    assert event.payload["actions"][0]["action_id"] == "send_to_customer"
    assert event.payload["response_url"].startswith("https://hooks.slack.com/")


def test_two_clicks_of_one_button_get_their_own_dedupe_keys():
    def click(action_ts: str):
        body = _interaction_body(_block_action("approve", action_ts=action_ts))
        return parse(_headers(body), body)

    assert click("2.0").dedupe_key() != click("3.0").dedupe_key()
    assert click("2.0").dedupe_key() == click("2.0").dedupe_key()


def test_a_message_shortcut_registers_as_its_callback_id():
    body = _interaction_body(
        {
            "type": "message_action",
            "callback_id": "escalate",
            "trigger_id": "13345224609.738474920.8088930838d88f008e0",
            "action_ts": "1700000002.000000",
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "message": {"ts": "1700000000.000100"},
        }
    )
    assert parse(_headers(body), body).qualified_type == "message_action.escalate"


def test_a_slash_command_registers_as_its_name():
    body = urlencode(
        {
            "token": "legacy",
            "command": "/hi-agent",
            "text": "[acme] scale up the m6a nodes",
            "channel_id": "C1",
            "user_id": "U1",
            "trigger_id": "13345224609.738474920.8088930838d88f008e0",
        }
    ).encode()
    event = parse(_headers(body), body)
    assert event.qualified_type == "command.hi-agent"
    assert event.event_type == events.Command.ANY
    assert event.scope == "C1"
    assert event.actor == "U1"
    assert event.payload["text"] == "[acme] scale up the m6a nodes"
    assert event.dedupe_key()  # no resource; keyed by trigger_id


def test_the_signature_covers_form_encoded_bodies_too():
    body = urlencode({"command": "/hi-agent", "text": "hello"}).encode()
    assert verify(body, _headers(body), SECRET) is True
    assert verify(body, _headers(body), "wrong-secret") is False


def test_the_ssl_check_probe_is_answered_and_deliveries_are_not():
    assert SlackProvider().handshake({}, b"token=abc&ssl_check=1") == {"ok": True}
    # Ordinary form deliveries proceed to verification, not the handshake.
    assert SlackProvider().handshake({}, _interaction_body(_block_action("approve"))) is None
    assert SlackProvider().handshake({}, b"command=%2Fhi-agent&text=hello") is None
