"""Jira-specific verification and normalization, beyond conformance."""

from __future__ import annotations

import json

from flyteplugins.jira import JiraProvider, events, parse, verify

TOKEN = "jira-token"


def _parse(payload: dict):
    body = json.dumps(payload).encode()
    return parse({"X-Webhook-Token": TOKEN}, body)


def test_the_shared_token_is_compared_not_a_signature():
    assert verify(b"anything", {"X-Webhook-Token": TOKEN}, TOKEN) is True
    assert verify(b"anything", {"X-Webhook-Token": "wrong"}, TOKEN) is False
    assert verify(b"anything", {}, TOKEN) is False


def test_the_provider_declares_itself_unsigned():
    """Jira does not sign; the dashboard says so rather than implying otherwise."""
    assert JiraProvider().signed is False


def test_the_webhook_event_name_is_the_qualified_type():
    assert (
        _parse({"webhookEvent": "jira:issue_created", "issue": {"key": "PROJ-1", "fields": {}}}).qualified_type
        == events.Issue.CREATED
    )


def test_comment_events_carry_no_jira_prefix():
    """That inconsistency is Jira's; the constants carry the exact wire values."""
    assert (
        _parse({"webhookEvent": "comment_created", "issue": {"key": "PROJ-1", "fields": {}}}).qualified_type
        == events.Comment.CREATED
    )


def test_the_project_key_comes_from_issue_fields():
    event = _parse(
        {
            "webhookEvent": "jira:issue_created",
            "issue": {"key": "PROJ-1", "fields": {"project": {"key": "PROJ"}, "summary": "A bug"}},
        }
    )
    assert event.scope == "PROJ"
    assert event.resource_id == "PROJ-1"
