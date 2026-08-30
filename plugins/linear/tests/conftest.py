"""Shared fixtures for Linear plugin tests."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
import respx

GRAPHQL_URL = "https://api.linear.app/graphql"


@pytest.fixture
def linear_api():
    """A respx router; register routes against the GraphQL endpoint."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")
    return "lin_api_test"


@pytest.fixture
def webhook_secret(monkeypatch):
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "linear-secret")
    return "linear-secret"


def sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def issue_payload(action: str = "create", team_id: str = "team-1", title: str = "A bug") -> dict:
    return {
        "action": action,
        "type": "Issue",
        "url": "https://linear.app/acme/issue/ENG-42",
        "createdAt": "2024-01-01T00:00:00Z",
        "webhookId": "wh-1",
        "organization": {"id": "org-1", "name": "acme"},
        "data": {
            "id": "issue-uuid",
            "title": title,
            "teamId": team_id,
            "stateId": "state-1",
            "url": "https://linear.app/acme/issue/ENG-42",
        },
    }


def webhook_headers(body: bytes, secret: str) -> dict:
    return {
        "X-Linear-Signature": sign(body, secret),
        "Content-Type": "application/json",
    }


def webhook_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
