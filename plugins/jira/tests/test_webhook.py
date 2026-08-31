"""Tests for Jira webhook token verification and event parsing."""

from __future__ import annotations

from conftest import json_body, webhook_payload

from flyteplugins.jira import parse_webhook, verify_webhook_token
from flyteplugins.jira._errors import WebhookSignatureError
from flyteplugins.jira._webhook import issue_from_payload


def test_verify_webhook_token():
    assert verify_webhook_token("s", "s") is True
    assert verify_webhook_token("wrong", "s") is False
    assert verify_webhook_token(None, "s") is False


def test_parse_issue_event():
    payload = webhook_payload(event="jira:issue_created", key="PROJ-9")
    event = parse_webhook({}, json_body(payload))
    assert event.webhook_event == "jira:issue_created"
    assert event.qualified_type == "jira:issue_created"
    assert event.issue_key == "PROJ-9"
    assert event.summary == "Fix the bug"
    assert event.status == "To Do"
    assert event.issue_type == "Bug"
    assert event.project_key == "PROJ"
    assert event.actor == "Bob"


def test_dedupe_key_stable_per_delivery():
    body = json_body(webhook_payload())
    e1 = parse_webhook({}, body)
    e2 = parse_webhook({}, body)
    assert e1.dedupe_key() == e2.dedupe_key()

    payload2 = webhook_payload()
    payload2["timestamp"] = 1700000000001
    e3 = parse_webhook({}, json_body(payload2))
    assert e3.dedupe_key() != e1.dedupe_key()


def test_issue_from_payload():
    issue = issue_from_payload(webhook_payload(key="PROJ-5"))
    assert issue["key"] == "PROJ-5"
    assert issue["description"] == "It broke."
    assert issue_from_payload({"webhookEvent": "x"}) is None


def test_parse_invalid_json_raises():
    try:
        parse_webhook({}, b"not json")
        raise AssertionError("expected WebhookSignatureError")
    except WebhookSignatureError:
        pass


def test_non_ascii_token_header_is_rejected_not_raised():
    """An attacker-controlled header must return False, not raise TypeError."""
    assert verify_webhook_token("üüü", "secret") is False
