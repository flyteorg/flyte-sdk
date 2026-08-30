"""Tests for the ClickUp REST API client."""

from __future__ import annotations

import httpx
import pytest

from flyteplugins.clickup import ClickUpAPIError, ClickUpClient, MissingCredentialsError

TASK_JSON = {
    "id": "t1",
    "name": "Fix the thing",
    "description": "details",
    "status": {"status": "to do"},
    "priority": {"priority": "2"},
    "url": "https://app.clickup.com/t/t1",
    "list": {"id": "l1"},
    "assignees": [{"username": "amy"}],
    "tags": [{"name": "bug"}],
    "date_created": "1",
    "date_updated": "2",
}


async def test_get_task_simplified(clickup_api):
    clickup_api.get("/task/t1").respond(json=TASK_JSON)
    async with ClickUpClient(token="k") as client:
        task = await client.get_task("t1")
    assert task["id"] == "t1"
    assert task["status"] == "to do"
    assert task["assignees"] == ["amy"]
    assert task["tags"] == ["bug"]


async def test_auth_header_is_raw_token(clickup_api):
    route = clickup_api.get("/user").respond(json={"user": {"id": 1, "username": "amy"}})
    async with ClickUpClient(token="k") as client:
        await client.get_user()
    assert route.calls[0].request.headers["Authorization"] == "k"
    assert route.calls[0].request.headers["ClickUp-Client"] == "flyteplugins-clickup"


async def test_missing_token(monkeypatch):
    monkeypatch.delenv("CLICKUP_TOKEN", raising=False)
    with pytest.raises(MissingCredentialsError) as excinfo:
        async with ClickUpClient():
            pass
    assert "CLICKUP_TOKEN" in str(excinfo.value)


async def test_list_statuses(clickup_api):
    clickup_api.get("/list/l1").respond(
        json={"id": "l1", "statuses": [{"status": "to do"}, {"status": "in progress"}, {"status": "done"}]}
    )
    async with ClickUpClient(token="k") as client:
        statuses = await client.list_statuses("l1")
    assert statuses == ["to do", "in progress", "done"]


async def test_list_tasks_status_filter(clickup_api):
    route = clickup_api.get("/list/l1/task").respond(json={"tasks": [TASK_JSON]})
    async with ClickUpClient(token="k") as client:
        tasks = await client.list_tasks("l1", statuses=["to do"])
    assert tasks[0]["id"] == "t1"
    assert route.calls[0].request.url.params["statuses[]"] == "to do"


async def test_create_task_payload(clickup_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=TASK_JSON)

    clickup_api.post("/list/l1/task").mock(side_effect=capture)
    async with ClickUpClient(token="k") as client:
        task = await client.create_task("l1", "Fix the thing", description="details", priority=2)
    assert task["id"] == "t1"
    assert captured["body"] == {"name": "Fix the thing", "description": "details", "priority": 2}


async def test_update_task_status(clickup_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=TASK_JSON)

    clickup_api.put("/task/t1").mock(side_effect=capture)
    async with ClickUpClient(token="k") as client:
        await client.update_task("t1", status="done")
    assert captured["body"] == {"status": "done"}


async def test_add_comment(clickup_api):
    clickup_api.post("/task/t1/comment").respond(json={"id": "c1"})
    async with ClickUpClient(token="k") as client:
        comment = await client.add_comment("t1", "working on it")
    assert comment == {"id": "c1"}


async def test_delete_task(clickup_api):
    route = clickup_api.delete("/task/t1").respond(status_code=200, content=b"")
    async with ClickUpClient(token="k") as client:
        assert await client.delete_task("t1") is None
    assert route.called


async def test_api_error_message(clickup_api):
    clickup_api.get("/task/nope").respond(status_code=404, json={"err": "Not Found"})
    async with ClickUpClient(token="k") as client:
        with pytest.raises(ClickUpAPIError) as excinfo:
            await client.get_task("nope")
    assert excinfo.value.status_code == 404
    assert "Not Found" in str(excinfo.value)


async def test_retries_on_429(clickup_api):
    route = clickup_api.get("/task/t1")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=TASK_JSON),
    ]
    from flyteplugins.clickup import Config

    async with ClickUpClient(Config(retry_backoff=0.0), token="k") as client:
        task = await client.get_task("t1")
    assert task["id"] == "t1"
    assert route.call_count == 2


async def test_list_lists_requires_scope():
    async with ClickUpClient(token="k") as client:
        async with client:
            with pytest.raises(ValueError):
                await client.list_lists()
