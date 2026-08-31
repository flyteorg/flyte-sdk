"""Tests for ClickUp webhook signature verification and event parsing."""

from __future__ import annotations

from conftest import sign, task_payload, webhook_body, webhook_headers

from flyteplugins.clickup import parse_webhook, verify_webhook_signature
from flyteplugins.clickup._errors import WebhookSignatureError


def test_verify_signature():
    body = b'{"event": "taskCreated"}'
    assert verify_webhook_signature(body, sign(body, "s"), "s") is True
    assert verify_webhook_signature(body, "deadbeef", "s") is False
    assert verify_webhook_signature(body, None, "s") is False


def test_parse_task_event():
    payload = task_payload(event="taskStatusUpdated", task_id="t9", list_id="l3", status="done")
    body = webhook_body(payload)
    event = parse_webhook(webhook_headers(body, "s"), body)
    assert event.event == "taskStatusUpdated"
    assert event.qualified_type == "taskStatusUpdated"
    assert event.task_id == "t9"
    assert event.list_id == "l3"
    assert event.task_name == "Fix the thing"
    assert event.task_status == "done"
    assert event.task_url == "https://app.clickup.com/t/t9"
    assert event.event_timestamp == 1700000000000


def test_dedupe_key_changes_with_timestamp():
    body1 = webhook_body(task_payload())
    e1 = parse_webhook(webhook_headers(body1, "s"), body1)

    # same delivery retried: identical key
    e2 = parse_webhook(webhook_headers(body1, "s"), body1)
    assert e1.dedupe_key() == e2.dedupe_key()

    # a later update to the same task has a different timestamp -> new key
    payload2 = task_payload()
    payload2["timestamp"] = 1700000000001
    body2 = webhook_body(payload2)
    e3 = parse_webhook(webhook_headers(body2, "s"), body2)
    assert e3.dedupe_key() != e1.dedupe_key()


def test_parse_invalid_json_raises():
    try:
        parse_webhook({}, b"not json")
        raise AssertionError("expected WebhookSignatureError")
    except WebhookSignatureError:
        pass


def test_list_id_falls_back_to_the_nested_task():
    """Task-scoped events carry the list id only on the nested task."""
    payload = task_payload()
    del payload["list_id"]
    payload["task"]["list"] = {"id": "l7"}
    body = webhook_body(payload)
    event = parse_webhook(webhook_headers(body, "s"), body)
    assert event.list_id == "l7"


def test_non_ascii_signature_header_is_rejected_not_raised():
    assert verify_webhook_signature(b"{}", "üüü", "secret") is False
