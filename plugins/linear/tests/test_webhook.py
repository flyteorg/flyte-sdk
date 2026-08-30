"""Tests for Linear webhook signature verification and event parsing."""

from __future__ import annotations

from conftest import issue_payload, webhook_body, webhook_headers

from flyteplugins.linear import parse_webhook, verify_webhook_signature
from flyteplugins.linear._errors import WebhookSignatureError


def test_verify_signature():
    body = b'{"action": "create"}'
    from conftest import sign

    assert verify_webhook_signature(body, sign(body, "s"), "s") is True
    assert verify_webhook_signature(body, "deadbeef", "s") is False
    assert verify_webhook_signature(body, None, "s") is False


def test_parse_issue_event():
    payload = issue_payload(action="create", team_id="team-1", title="A bug")
    body = webhook_body(payload)
    event = parse_webhook(webhook_headers(body, "s"), body)
    assert event.entity_type == "Issue"
    assert event.action == "create"
    assert event.qualified_type == "Issue.create"
    assert event.entity_id == "issue-uuid"
    assert event.title == "A bug"
    assert event.team_id == "team-1"
    assert event.state_id == "state-1"
    assert event.organization == "acme"
    assert event.entity_url == "https://linear.app/acme/issue/ENG-42"


def test_dedupe_key_stable():
    body = webhook_body(issue_payload())
    e1 = parse_webhook(webhook_headers(body, "s"), body)
    e2 = parse_webhook(webhook_headers(body, "s"), body)
    assert e1.dedupe_key() == e2.dedupe_key()


def test_parse_invalid_json_raises():
    try:
        parse_webhook({}, b"not json")
        raise AssertionError("expected WebhookSignatureError")
    except WebhookSignatureError:
        pass
