"""Shared fixtures for Jira plugin tests."""

from __future__ import annotations

import json

import pytest
import respx

SITE = "https://acme.atlassian.net"
API_BASE = f"{SITE}/rest/api/3"


@pytest.fixture
def jira_api():
    """A respx router mocking the Jira REST API."""
    with respx.mock(base_url=API_BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", SITE)
    monkeypatch.setenv("JIRA_EMAIL", "bot@acme.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "jira-token")
    return {"base_url": SITE, "email": "bot@acme.com", "api_token": "jira-token"}


@pytest.fixture
def webhook_token(monkeypatch):
    monkeypatch.setenv("JIRA_WEBHOOK_TOKEN", "wh-secret")
    return "wh-secret"


def issue_json(key: str = "PROJ-1", summary: str = "Fix the bug", status: str = "To Do") -> dict:
    return {
        "id": "10001",
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": status},
            "issuetype": {"name": "Bug"},
            "assignee": {"displayName": "Amy"},
            "reporter": {"displayName": "Bob"},
            "priority": {"name": "High"},
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": "It broke."}]}],
            },
            "labels": ["regression"],
            "project": {"key": "PROJ"},
            "created": "2024-01-01T00:00:00.000+0000",
            "updated": "2024-01-02T00:00:00.000+0000",
        },
    }


def webhook_payload(event: str = "jira:issue_created", key: str = "PROJ-1") -> dict:
    return {
        "webhookEvent": event,
        "timestamp": 1700000000000,
        "user": {"displayName": "Bob"},
        "issue": issue_json(key=key),
    }


def json_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
