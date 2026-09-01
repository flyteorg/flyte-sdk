"""Tests for the Linear GraphQL client."""

from __future__ import annotations

import httpx
import pytest
from conftest import GRAPHQL_URL

from flyteplugins.linear import LinearAPIError, LinearClient, MissingCredentialsError

ISSUE_NODE = {
    "id": "uuid-1",
    "identifier": "ENG-42",
    "title": "Fix the bug",
    "description": "It broke",
    "url": "https://linear.app/acme/issue/ENG-42",
    "priority": 2,
    "createdAt": "2024-01-01",
    "updatedAt": "2024-01-02",
    "state": {"id": "s1", "name": "In Progress", "type": "started"},
    "assignee": {"id": "u1", "name": "amy", "displayName": "Amy"},
    "team": {"id": "t1", "key": "ENG", "name": "Engineering"},
    "labels": {"nodes": [{"id": "l1", "name": "bug"}]},
}


def graphql_route(router, data=None, errors=None, status_code=200):
    return router.post(GRAPHQL_URL).respond(status_code=status_code, json={"data": data, "errors": errors})


async def test_get_viewer(linear_api):
    graphql_route(linear_api, data={"viewer": {"id": "u1", "name": "amy", "displayName": "Amy", "email": "a@x"}})
    async with LinearClient(api_key="k") as client:
        viewer = await client.get_viewer.aio()
    assert viewer == {"id": "u1", "name": "Amy", "email": "a@x"}


async def test_list_teams(linear_api):
    graphql_route(linear_api, data={"teams": {"nodes": [{"id": "t1", "key": "ENG", "name": "Engineering"}]}})
    async with LinearClient(api_key="k") as client:
        teams = await client.list_teams.aio()
    assert teams == [{"id": "t1", "key": "ENG", "name": "Engineering"}]


async def test_list_issues_builds_filter(linear_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"data": {"issues": {"nodes": [ISSUE_NODE]}}})

    linear_api.post(GRAPHQL_URL).mock(side_effect=capture)
    async with LinearClient(api_key="k") as client:
        issues = await client.list_issues.aio(team_key="ENG", state="In Progress")
    assert issues[0]["identifier"] == "ENG-42"
    assert issues[0]["state"] == "In Progress"
    assert issues[0]["labels"] == ["bug"]
    variables = captured["body"]["variables"]
    assert variables["filter"] == {
        "and": [{"team": {"key": {"eq": "ENG"}}}, {"state": {"name": {"eq": "In Progress"}}}]
    }


async def test_get_issue_by_identifier(linear_api):
    graphql_route(linear_api, data={"issues": {"nodes": [ISSUE_NODE]}})
    async with LinearClient(api_key="k") as client:
        issue = await client.get_issue.aio("ENG-42")
    assert issue["identifier"] == "ENG-42"
    assert issue["team"] == "ENG"


async def test_get_issue_not_found(linear_api):
    graphql_route(linear_api, data={"issues": {"nodes": []}})
    async with LinearClient(api_key="k") as client:
        with pytest.raises(LinearAPIError) as excinfo:
            await client.get_issue.aio("ENG-999")
    assert "not found" in str(excinfo.value)


async def test_graphql_error_raised(linear_api):
    graphql_route(linear_api, data=None, errors=[{"message": "unauthorized"}])
    async with LinearClient(api_key="k") as client:
        with pytest.raises(LinearAPIError) as excinfo:
            await client.get_viewer.aio()
    assert "unauthorized" in str(excinfo.value)


async def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError) as excinfo:
        async with LinearClient():
            pass
    assert "LINEAR_API_KEY" in str(excinfo.value)


async def test_create_issue(linear_api):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"data": {"issueCreate": {"success": True, "issue": ISSUE_NODE}}})

    linear_api.post(GRAPHQL_URL).mock(side_effect=capture)
    async with LinearClient(api_key="k") as client:
        issue = await client.create_issue.aio("t1", "New issue", description="details", priority=1)
    assert issue["identifier"] == "ENG-42"
    issue_input = captured["body"]["variables"]["input"]
    assert issue_input == {"teamId": "t1", "title": "New issue", "description": "details", "priority": 1}


async def test_update_issue_state(linear_api):
    graphql_route(linear_api, data={"issueUpdate": {"success": True, "issue": ISSUE_NODE}})
    async with LinearClient(api_key="k") as client:
        issue = await client.update_issue.aio("uuid-1", state_id="s2")
    assert issue["id"] == "uuid-1"


async def test_add_comment(linear_api):
    graphql_route(
        linear_api,
        data={"commentCreate": {"success": True, "comment": {"id": "c1", "url": "u", "createdAt": "t"}}},
    )
    async with LinearClient(api_key="k") as client:
        comment = await client.add_comment.aio("uuid-1", "fixed in abc123")
    assert comment["id"] == "c1"


async def test_retries_on_500(linear_api):
    route = linear_api.post(GRAPHQL_URL)
    route.side_effect = [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"data": {"teams": {"nodes": []}}}),
    ]
    from flyteplugins.linear import Config

    async with LinearClient(Config(retry_backoff=0.0), api_key="k") as client:
        teams = await client.list_teams.aio()
    assert teams == []
    assert route.call_count == 2


def test_the_blocking_call_form_works_outside_an_event_loop(linear_api):
    """`with Client() as c: c.method(...)` -- the point of syncifying the client.

    `__enter__` runs `__aenter__` on syncify's background loop, the same loop the
    syncified methods run on, so the httpx client is created and used on one loop.
    """
    linear_api.post(GRAPHQL_URL).respond(
        json={"data": {"viewer": {"id": "u1", "displayName": "Ada", "email": "a@b.c"}}}
    )
    with LinearClient(api_key="k") as client:
        issue = client.get_viewer()
    assert issue["name"] == "Ada"


async def test_the_async_form_is_the_same_method_via_aio(linear_api):
    """Both call forms are the same method: `m(...)` blocks, `await m.aio(...)` does not."""
    linear_api.post(GRAPHQL_URL).respond(
        json={"data": {"viewer": {"id": "u1", "displayName": "Ada", "email": "a@b.c"}}}
    )
    async with LinearClient(api_key="k") as client:
        issue = await client.get_viewer.aio()
    assert issue["name"] == "Ada"


def test_methods_expose_both_call_forms():
    from flyte.syncify import syncify  # noqa: F401

    method = LinearClient.get_viewer
    assert hasattr(method, "aio"), "syncified methods must offer an async form"
