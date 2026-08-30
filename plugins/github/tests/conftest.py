"""Shared fixtures for GitHub plugin tests."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
import respx


@pytest.fixture
def github_api():
    """A respx router mocking https://api.github.com."""
    with respx.mock(base_url="https://api.github.com", assert_all_called=False) as router:
        yield router


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    return "test-token"


@pytest.fixture
def webhook_secret(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    return "test-secret"


def sign(payload: bytes, secret: str) -> str:
    """Compute the X-Hub-Signature-256 header value for a payload."""
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def pr_payload(number: int = 42, action: str = "opened", repo: str = "octo/repo") -> dict:
    return {
        "action": action,
        "number": number,
        "pull_request": {
            "number": number,
            "title": f"PR #{number}",
            "html_url": f"https://github.com/{repo}/pull/{number}",
        },
        "repository": {"full_name": repo},
        "sender": {"login": "octocat"},
    }


def webhook_headers(body: bytes, secret: str, event: str = "pull_request", delivery: str = "d-1") -> dict:
    return {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": sign(body, secret),
        "Content-Type": "application/json",
    }


def webhook_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
