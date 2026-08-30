"""Tests for the Notion API client."""

from __future__ import annotations

import httpx
import pytest
from conftest import page_json, query_response

from flyteplugins.notion import MissingCredentialsError, NotionAPIError, NotionClient, title_property


async def test_get_me(notion_api):
    notion_api.get("/users/me").respond(json={"id": "bot1", "name": "Flyte Bot", "type": "bot"})
    async with NotionClient(token="k") as client:
        me = await client.get_me()
    assert me == {"id": "bot1", "name": "Flyte Bot", "type": "bot"}


async def test_headers(notion_api):
    route = notion_api.get("/users/me").respond(json={"id": "b", "name": "n", "type": "bot"})
    async with NotionClient(token="k") as client:
        await client.get_me()
    headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer k"
    assert headers["Notion-Version"] == "2022-06-28"


async def test_missing_token(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    with pytest.raises(MissingCredentialsError) as excinfo:
        async with NotionClient():
            pass
    assert "NOTION_TOKEN" in str(excinfo.value)


async def test_search_object_filter(notion_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"results": [page_json()]})

    notion_api.post("/search").mock(side_effect=capture)
    async with NotionClient(token="k") as client:
        results = await client.search("roadmap", object_type="page")
    assert captured["body"]["filter"] == {"property": "object", "value": "page"}
    assert results[0]["title"] == "Roadmap item"
    assert results[0]["object"] == "page"


async def test_get_page_extracts_title(notion_api):
    notion_api.get("/pages/p1").respond(json=page_json(title="My page"))
    async with NotionClient(token="k") as client:
        page = await client.get_page("p1")
    assert page["title"] == "My page"
    assert page["parent_type"] == "database_id"


async def test_query_database_since_filter(notion_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=query_response([page_json()]))

    notion_api.post("/databases/db1/query").mock(side_effect=capture)
    async with NotionClient(token="k") as client:
        pages = await client.query_database_since("db1", "2024-05-01T00:00:00.000Z")
    assert pages[0]["id"] == "p1"
    assert captured["body"]["filter"] == {
        "timestamp": "last_edited_time",
        "last_edited_time": {"after": "2024-05-01T00:00:00.000Z"},
    }
    assert captured["body"]["sorts"] == [{"timestamp": "last_edited_time", "direction": "ascending"}]


async def test_create_database_page(notion_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=page_json())

    notion_api.post("/pages").mock(side_effect=capture)
    async with NotionClient(token="k") as client:
        page = await client.create_database_page("db1", {"Name": title_property("New row")})
    assert page["id"] == "p1"
    assert captured["body"]["parent"] == {"database_id": "db1"}
    assert captured["body"]["properties"]["Name"]["title"][0]["text"]["content"] == "New row"


async def test_create_child_page_title_nesting(notion_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=page_json())

    notion_api.post("/pages").mock(side_effect=capture)
    async with NotionClient(token="k") as client:
        await client.create_page("parent-page", title="Child")
    assert captured["body"]["parent"] == {"page_id": "parent-page"}
    assert captured["body"]["properties"] == {"title": {"title": [{"text": {"content": "Child"}}]}}


async def test_update_page_archives(notion_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=page_json())

    notion_api.patch("/pages/p1").mock(side_effect=capture)
    async with NotionClient(token="k") as client:
        await client.archive_page("p1")
    assert captured["body"] == {"archived": True}


async def test_append_blocks(notion_api):
    notion_api.patch("/blocks/p1/children").respond(json={"results": [{"id": "b1"}]})
    async with NotionClient(token="k") as client:
        blocks = await client.append_blocks("p1", [{"object": "block"}])
    assert blocks == [{"id": "b1"}]


async def test_api_error_code(notion_api):
    notion_api.get("/pages/nope").respond(
        status_code=404, json={"object": "error", "status": 404, "code": "object_not_found", "message": "not found"}
    )
    async with NotionClient(token="k") as client:
        with pytest.raises(NotionAPIError) as excinfo:
            await client.get_page("nope")
    assert excinfo.value.code == "object_not_found"
    assert excinfo.value.status_code == 404


async def test_retries_on_429(notion_api):
    route = notion_api.get("/users/me")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"id": "b", "name": "n", "type": "bot"}),
    ]
    from flyteplugins.notion import Config

    async with NotionClient(Config(retry_backoff=0.0), token="k") as client:
        await client.get_me()
    assert route.call_count == 2
