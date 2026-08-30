"""Shared fixtures for ClickUp plugin tests."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
import respx

API_BASE = "https://api.clickup.com/api/v2"


@pytest.fixture
def clickup_api():
    """A respx router mocking https://api.clickup.com/api/v2."""
    with respx.mock(base_url=API_BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("CLICKUP_TOKEN", "pk_test_token")
    return "pk_test_token"


@pytest.fixture
def webhook_secret(monkeypatch):
    monkeypatch.setenv("CLICKUP_WEBHOOK_SECRET", "cu-secret")
    return "cu-secret"


def sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def task_payload(event: str = "taskCreated", task_id: str = "t1", list_id: str = "l1", status: str = "to do") -> dict:
    return {
        "event": event,
        "task_id": task_id,
        "list_id": list_id,
        "webhook_id": "wh-1",
        "timestamp": 1700000000000,
        "task": {
            "id": task_id,
            "name": "Fix the thing",
            "url": f"https://app.clickup.com/t/{task_id}",
            "status": {"status": status},
        },
    }


def webhook_headers(body: bytes, secret: str) -> dict:
    return {
        "x-clickup-signature": sign(body, secret),
        "Content-Type": "application/json",
    }


def webhook_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
