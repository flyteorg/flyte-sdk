"""Tests for webhook signature verification and event parsing."""

from __future__ import annotations

from conftest import pr_payload, webhook_body, webhook_headers

from flyteplugins.github import parse_webhook, verify_webhook_signature
from flyteplugins.github._errors import WebhookSignatureError


def test_verify_signature_valid():
    body = b'{"action": "opened"}'
    header = "sha256=" + __import__("hmac").new(b"secret", body, "sha256").hexdigest()
    assert verify_webhook_signature(body, header, "secret") is True


def test_verify_signature_invalid():
    body = b'{"action": "opened"}'
    assert verify_webhook_signature(body, "sha256=deadbeef", "secret") is False
    assert verify_webhook_signature(body, None, "secret") is False
    assert verify_webhook_signature(body, "md5=abc", "secret") is False


def test_parse_pull_request_event():
    payload = pr_payload(number=7, action="opened", repo="octo/repo")
    body = webhook_body(payload)
    event = parse_webhook(webhook_headers(body, "s", event="pull_request"), body)
    assert event.event_type == "pull_request"
    assert event.action == "opened"
    assert event.qualified_type == "pull_request.opened"
    assert event.repository == "octo/repo"
    assert event.number == 7
    assert event.sender == "octocat"
    assert event.title == "PR #7"
    assert event.delivery_id == "d-1"


def test_parse_issue_comment_event_uses_issue_number():
    payload = {
        "action": "created",
        "issue": {"number": 99, "title": "A bug", "html_url": "https://github.com/octo/repo/issues/99"},
        "comment": {"id": 555, "body": "hi", "html_url": "https://github.com/octo/repo/issues/99#issuecomment-555"},
        "repository": {"full_name": "octo/repo"},
        "sender": {"login": "someone"},
    }
    body = webhook_body(payload)
    event = parse_webhook(webhook_headers(body, "s", event="issue_comment"), body)
    assert event.number == 99  # issue number takes precedence over comment id
    assert event.url and "#issuecomment" in event.url


def test_dedupe_key_stable_across_redeliveries():
    payload = pr_payload(number=7, repo="octo/repo")
    body = webhook_body(payload)
    e1 = parse_webhook(webhook_headers(body, "s", delivery="a"), body)
    e2 = parse_webhook(webhook_headers(body, "s", delivery="b"), body)
    assert e1.dedupe_key() == e2.dedupe_key()

    # events without a number fall back to the delivery id
    push_body = webhook_body({"repository": {"full_name": "octo/repo"}})
    push = parse_webhook(webhook_headers(push_body, "s", event="push", delivery="x"), push_body)
    assert push.dedupe_key() != e1.dedupe_key()


def test_parse_missing_event_header_raises():
    from flyteplugins.github._errors import WebhookSignatureError as WSE

    try:
        parse_webhook({"X-GitHub-Delivery": "1"}, b"{}")
        raise AssertionError("expected WebhookSignatureError")
    except WSE:
        pass


def test_parse_invalid_json_raises():
    try:
        parse_webhook({"X-GitHub-Event": "push"}, b"not json")
        raise AssertionError("expected WebhookSignatureError")
    except WebhookSignatureError:
        pass
