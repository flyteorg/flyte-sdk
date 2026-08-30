"""Shared fixtures for Notion plugin tests."""

from __future__ import annotations

import json

import pytest
import respx

API_BASE = "https://api.notion.com/v1"


@pytest.fixture
def notion_api():
    """A respx router mocking https://api.notion.com/v1."""
    with respx.mock(base_url=API_BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test")
    return "ntn_test"


@pytest.fixture
def poll_token(monkeypatch):
    monkeypatch.setenv("NOTION_POLL_TOKEN", "poll-secret")
    return "poll-secret"


def page_json(page_id: str = "p1", title: str = "Roadmap item", edited: str = "2024-06-01T00:00:00.000Z") -> dict:
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "archived": False,
        "created_time": "2024-05-01T00:00:00.000Z",
        "last_edited_time": edited,
        "parent": {"type": "database_id", "database_id": "db1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": title}]},
        },
    }


def query_response(pages: list[dict]) -> dict:
    return {"object": "list", "results": pages, "has_more": False, "next_cursor": None}


def json_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()
