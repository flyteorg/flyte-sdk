"""Tests for Slack Events API signature verification and parsing."""

from __future__ import annotations

import time

from conftest import SIGNING_SECRET, event_body, event_headers, message_event, sign

from flyteplugins.slack import parse_event, parse_url_verification, verify_event_signature
from flyteplugins.slack._errors import EventSignatureError


def test_verify_signature_valid():
    body = b'{"a": 1}'
    ts = int(time.time())
    assert verify_event_signature(body, str(ts), sign(body, SIGNING_SECRET, ts), SIGNING_SECRET) is True


def test_verify_signature_rejects_invalid_or_missing():
    body = b'{"a": 1}'
    ts = str(int(time.time()))
    assert verify_event_signature(body, ts, "v0=deadbeef", SIGNING_SECRET) is False
    assert verify_event_signature(body, ts, None, SIGNING_SECRET) is False
    assert verify_event_signature(body, None, "v0=abc", SIGNING_SECRET) is False


def test_verify_signature_rejects_stale_timestamp():
    body = b'{"a": 1}'
    ts = int(time.time()) - 3600  # one hour old
    assert verify_event_signature(body, str(ts), sign(body, SIGNING_SECRET, ts), SIGNING_SECRET) is False


def test_verify_signature_injectable_clock():
    body = b'{"a": 1}'
    ts = 1000
    signature = sign(body, SIGNING_SECRET, ts)
    assert verify_event_signature(body, str(ts), signature, SIGNING_SECRET, now=1200) is True
    assert verify_event_signature(body, str(ts), signature, SIGNING_SECRET, now=9999) is False


def test_parse_url_verification():
    assert parse_url_verification(b'{"type": "url_verification", "challenge": "abc"}') == "abc"
    assert parse_url_verification(b'{"type": "event_callback"}') is None
    assert parse_url_verification(b"not json") is None


def test_parse_message_event():
    payload = message_event(channel="C123", ts="1.0", user="U42", text="hello world")
    body = event_body(payload)
    event = parse_event(event_headers(body), body)
    assert event.event_type == "message"
    assert event.qualified_type == "message"
    assert event.channel == "C123"
    assert event.ts == "1.0"
    assert event.user == "U42"
    assert event.text == "hello world"
    assert event.event_id == "Ev123"
    assert event.team_id == "T1"


def test_thread_events_collapse_to_root_for_dedupe():
    root = message_event(ts="1.0")
    reply = message_event(ts="2.0")
    reply["event"]["thread_ts"] = "1.0"
    e_root = parse_event({}, event_body(root))
    e_reply = parse_event({}, event_body(reply))
    assert e_reply.root_ts == "1.0"
    assert e_reply.dedupe_key() == e_root.dedupe_key()


def test_parse_reaction_event():
    payload = {
        "type": "event_callback",
        "event": {"type": "reaction_added", "reaction": "bug", "user": "U1", "item": {"channel": "C1"}},
    }
    event = parse_event({}, event_body(payload))
    assert event.event_type == "reaction_added"
    assert event.reaction == "bug"


def test_parse_event_missing_event_object_raises():
    try:
        parse_event({}, b'{"type": "event_callback"}')
        raise AssertionError("expected EventSignatureError")
    except EventSignatureError:
        pass


def test_parse_invalid_json_raises():
    try:
        parse_event({}, b"not json")
        raise AssertionError("expected EventSignatureError")
    except EventSignatureError:
        pass
