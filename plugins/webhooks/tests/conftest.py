"""Shared fixtures for the webhooks plugin tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

GITHUB_SECRET = "gh-secret"
SLACK_SECRET = "slack-secret"
LINEAR_SECRET = "linear-secret"
CLICKUP_SECRET = "clickup-secret"
JIRA_TOKEN = "jira-token"


@pytest.fixture
def secrets(monkeypatch):
    """Mount every provider's secret."""
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", GITHUB_SECRET)
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SLACK_SECRET)
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", LINEAR_SECRET)
    monkeypatch.setenv("CLICKUP_WEBHOOK_SECRET", CLICKUP_SECRET)
    monkeypatch.setenv("JIRA_WEBHOOK_TOKEN", JIRA_TOKEN)


def body_of(payload: dict) -> bytes:
    return json.dumps(payload).encode()


def github_headers(body: bytes, event: str = "pull_request", delivery: str = "d-1") -> dict:
    sig = hmac.new(GITHUB_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"X-GitHub-Event": event, "X-GitHub-Delivery": delivery, "X-Hub-Signature-256": f"sha256={sig}"}


def slack_headers(body: bytes, timestamp: int | None = None) -> dict:
    ts = str(timestamp if timestamp is not None else int(time.time()))
    base = b"v0:" + ts.encode() + b":" + body
    sig = hmac.new(SLACK_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": f"v0={sig}"}


def linear_headers(body: bytes) -> dict:
    return {"X-Linear-Signature": hmac.new(LINEAR_SECRET.encode(), body, hashlib.sha256).hexdigest()}


def clickup_headers(body: bytes) -> dict:
    return {"X-Clickup-Signature": hmac.new(CLICKUP_SECRET.encode(), body, hashlib.sha256).hexdigest()}


def jira_headers(_body: bytes) -> dict:
    return {"X-Webhook-Token": JIRA_TOKEN}


def pr_payload(number: int = 7, action: str = "opened", repo: str = "octo/repo") -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "title": f"PR #{number}",
            "html_url": f"https://github.com/{repo}/pull/{number}",
            "updated_at": "2024-01-01T00:00:00Z",
        },
        "repository": {"full_name": repo},
        "sender": {"login": "octocat"},
    }


def slack_payload(event_type: str = "app_mention", channel: str = "C1", ts: str = "1.0") -> dict:
    return {
        "event_id": "Ev1",
        "event": {"type": event_type, "channel": channel, "ts": ts, "user": "U1", "text": "hey"},
    }


def linear_payload(action: str = "create", team_id: str = "team-1") -> dict:
    return {
        "action": action,
        "type": "Issue",
        "webhookId": "wh-1",
        "createdAt": "2024-01-01T00:00:00Z",
        "data": {"id": "issue-uuid", "title": "A bug", "teamId": team_id, "updatedAt": "2024-01-01T00:00:00Z"},
    }


def clickup_payload(event: str = "taskCreated", list_id: str = "l1") -> dict:
    return {
        "event": event,
        "task_id": "t1",
        "list_id": list_id,
        "webhook_id": "wh-1",
        "timestamp": 1700000000000,
        "task": {"id": "t1", "name": "Fix the thing", "url": "https://app.clickup.com/t/t1"},
    }


def jira_payload(event: str = "jira:issue_created", key: str = "PROJ-1") -> dict:
    return {
        "webhookEvent": event,
        "timestamp": 1700000000000,
        "user": {"displayName": "Bob"},
        "issue": {"key": key, "fields": {"summary": "A bug", "project": {"key": "PROJ"}}},
    }


#: (provider, header builder, payload builder, expected qualified_type)
ALL_PROVIDERS = [
    ("github", github_headers, pr_payload, "pull_request.opened"),
    ("slack", slack_headers, slack_payload, "app_mention"),
    ("linear", linear_headers, linear_payload, "Issue.create"),
    ("clickup", clickup_headers, clickup_payload, "taskCreated"),
    ("jira", jira_headers, jira_payload, "jira:issue_created"),
]
