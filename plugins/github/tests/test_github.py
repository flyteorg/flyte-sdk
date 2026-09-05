"""GitHub-specific normalization, beyond what conformance covers."""

from __future__ import annotations

import hashlib
import hmac
import json
from urllib.parse import urlencode

import pytest
from flyte.extras.webhooks import SignatureError

from flyteplugins.github import GitHubProvider, events, parse, verify

SECRET = "gh-secret"


def _headers(body: bytes, event: str = "pull_request", delivery: str = "d-1") -> dict:
    return {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
    }


def _parse(payload: dict, event: str = "pull_request") -> object:
    body = json.dumps(payload).encode()
    return parse(_headers(body, event=event), body)


def test_verify_requires_the_sha256_prefix():
    body = b"{}"
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert verify(body, {"X-Hub-Signature-256": f"sha256={digest}"}, SECRET) is True
    assert verify(body, {"X-Hub-Signature-256": digest}, SECRET) is False


def test_pull_request_normalizes_to_the_constant():
    event = _parse(
        {"action": "opened", "pull_request": {"number": 7, "title": "t"}, "repository": {"full_name": "octo/repo"}}
    )
    assert event.qualified_type == events.PullRequest.OPENED
    assert event.resource_id == "octo/repo#7"
    assert event.scope == "octo/repo"


def test_distinct_comments_on_one_issue_do_not_collapse():
    """Keyed on the issue alone, every comment after the first looks like a redelivery."""

    def comment(comment_id: int):
        return _parse(
            {
                "action": "created",
                "issue": {"number": 9, "title": "A bug"},
                "comment": {"id": comment_id},
                "repository": {"full_name": "octo/repo"},
            },
            event="issue_comment",
        )

    assert comment(1).dedupe_key() != comment(2).dedupe_key()
    assert comment(1).dedupe_key() == comment(1).dedupe_key()


def test_a_review_is_keyed_like_a_comment():
    event = _parse(
        {
            "action": "submitted",
            "pull_request": {"number": 7},
            "review": {"id": 4242},
            "repository": {"full_name": "octo/repo"},
        },
        event="pull_request_review",
    )
    assert event.resource_id == "octo/repo#7:4242"


def test_push_has_no_resource_and_falls_back_to_the_delivery_id():
    body = json.dumps({"repository": {"full_name": "octo/repo"}}).encode()
    a = parse(_headers(body, event="push", delivery="x"), body)
    b = parse(_headers(body, event="push", delivery="y"), body)
    assert a.resource_id is None
    assert a.dedupe_key() != b.dedupe_key()


def test_the_ping_handshake_is_answered():
    assert GitHubProvider().handshake({"X-GitHub-Event": "ping"}, b"{}") == {"ok": True, "ping": True}
    assert GitHubProvider().handshake({"X-GitHub-Event": "pull_request"}, b"{}") is None


#: The delivery shape of a real comment-triggered agent: someone comments a
#: command on an issue, the handler reads the command and the repo from the
#: payload. Modeled on the SWE-agent receiver in unionai-agents.
AGENT_TRIGGER_COMMENT = {
    "action": "created",
    "issue": {"number": 42, "title": "Fix the flaky test", "labels": [{"name": "bug"}]},
    "comment": {"id": 111, "body": "/swe_agent fix", "user": {"login": "octocat"}},
    "repository": {"full_name": "octo/repo", "html_url": "https://github.com/octo/repo"},
    "sender": {"login": "octocat"},
}


def _form_body(payload: dict) -> bytes:
    """The default-content-type delivery: the JSON under a `payload=` form field."""
    return urlencode({"payload": json.dumps(payload)}).encode()


def test_a_form_encoded_delivery_parses_like_its_json_twin():
    json_delivery = json.dumps(AGENT_TRIGGER_COMMENT).encode()
    form_delivery = _form_body(AGENT_TRIGGER_COMMENT)
    a = parse(_headers(json_delivery, event="issue_comment"), json_delivery)
    b = parse(_headers(form_delivery, event="issue_comment"), form_delivery)
    assert a.qualified_type == b.qualified_type == events.IssueComment.CREATED
    assert a.resource_id == b.resource_id == "octo/repo#42:111"
    # A redelivery on the other content type is still a redelivery.
    assert a.dedupe_key() == b.dedupe_key()
    # The handler sees the full payload either way — the trigger command, its
    # author, the labels — nothing the normalization drops.
    assert b.payload["comment"]["body"] == "/swe_agent fix"
    assert b.payload["issue"]["labels"] == [{"name": "bug"}]
    assert b.payload["repository"]["html_url"] == "https://github.com/octo/repo"


def test_the_signature_covers_form_encoded_bodies_too():
    body = _form_body(AGENT_TRIGGER_COMMENT)
    assert verify(body, _headers(body), SECRET) is True
    assert verify(body, _headers(body), "wrong-secret") is False


def test_a_form_delivery_without_a_payload_field_is_rejected():
    body = b"foo=bar"
    with pytest.raises(SignatureError):
        parse(_headers(body, event="issue_comment"), body)


def test_installation_events_key_on_the_delivery():
    """App webhooks receive these unconditionally; no repository, so no resource."""
    body = json.dumps({"action": "created", "installation": {"id": 1, "account": {"login": "octo"}}}).encode()
    a = parse(_headers(body, event="installation", delivery="d-a"), body)
    b = parse(_headers(body, event="installation", delivery="d-b"), body)
    assert a.qualified_type == events.Installation.CREATED
    assert a.resource_id is None
    assert a.dedupe_key() != b.dedupe_key()
