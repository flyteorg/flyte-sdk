"""Tests for the Jira REST API client."""

from __future__ import annotations

import base64

import httpx
import pytest
from conftest import SITE, issue_json

from flyteplugins.jira import JiraAPIError, JiraClient, MissingCredentialsError


async def test_basic_auth_header(jira_api, creds):
    route = jira_api.get("/myself").respond(json={"accountId": "a", "displayName": "Bot"})
    async with JiraClient() as client:
        await client.get_myself()
    header = route.calls[0].request.headers["Authorization"]
    assert header == "Basic " + base64.b64encode(b"bot@acme.com:jira-token").decode()


async def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("JIRA_BASE_URL", raising=False)
    with pytest.raises(MissingCredentialsError) as excinfo:
        async with JiraClient():
            pass
    assert "JIRA_BASE_URL" in str(excinfo.value)


async def test_get_issue_simplifies_and_extracts_description(jira_api, creds):
    jira_api.get("/issue/PROJ-1").respond(json=issue_json())
    async with JiraClient() as client:
        issue = await client.get_issue("PROJ-1")
    assert issue["key"] == "PROJ-1"
    assert issue["status"] == "To Do"
    assert issue["description"] == "It broke."
    assert issue["url"] == f"{SITE}/browse/PROJ-1"
    assert issue["labels"] == ["regression"]


async def test_search_issues(jira_api, creds):
    jira_api.get("/search").respond(json={"issues": [issue_json(), issue_json(key="PROJ-2")]})
    async with JiraClient() as client:
        issues = await client.search_issues("project = PROJ")
    assert [i["key"] for i in issues] == ["PROJ-1", "PROJ-2"]


async def test_create_issue_adf_description(jira_api, creds):
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(201, json={"id": "10001", "key": "PROJ-3"})

    jira_api.post("/issue").mock(side_effect=capture)
    async with JiraClient() as client:
        issue = await client.create_issue("PROJ", "New thing", description="details", priority="High")
    assert issue["key"] == "PROJ-3"
    fields = captured["body"]["fields"]
    assert fields["project"] == {"key": "PROJ"}
    assert fields["issuetype"] == {"name": "Task"}
    assert fields["description"]["content"][0]["content"][0]["text"] == "details"
    assert fields["priority"] == {"name": "High"}


async def test_transition_by_name(jira_api, creds):
    jira_api.get("/issue/PROJ-1/transitions").respond(
        json={"transitions": [{"id": "21", "name": "In Progress", "to": {"name": "In Progress"}}]}
    )
    captured = {}

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(204, content=b"")

    jira_api.post("/issue/PROJ-1/transitions").mock(side_effect=capture)
    async with JiraClient() as client:
        result = await client.transition_issue("PROJ-1", "in progress")
    assert result["transition"] == "21"
    assert captured["body"] == {"transition": {"id": "21"}}


async def test_transition_unknown_name_raises(jira_api, creds):
    jira_api.get("/issue/PROJ-1/transitions").respond(json={"transitions": [{"id": "21", "name": "In Progress"}]})
    async with JiraClient() as client:
        with pytest.raises(JiraAPIError) as excinfo:
            await client.transition_issue("PROJ-1", "Done")
    assert "In Progress" in str(excinfo.value)


async def test_add_comment(jira_api, creds):
    jira_api.post("/issue/PROJ-1/comment").respond(json={"id": "c1", "created": "t"})
    async with JiraClient() as client:
        comment = await client.add_comment("PROJ-1", "on it")
    assert comment == {"id": "c1", "created": "t"}


async def test_delete_issue(jira_api, creds):
    route = jira_api.delete("/issue/PROJ-1").respond(status_code=204, content=b"")
    async with JiraClient() as client:
        assert await client.delete_issue("PROJ-1") is None
    assert route.called


async def test_api_error_messages(jira_api, creds):
    jira_api.get("/issue/NOPE-1").respond(
        status_code=404, json={"errorMessages": ["Issue does not exist or you do not have permission to see it."]}
    )
    async with JiraClient() as client:
        with pytest.raises(JiraAPIError) as excinfo:
            await client.get_issue("NOPE-1")
    assert excinfo.value.status_code == 404
    assert "does not exist" in str(excinfo.value)


async def test_retries_on_429(jira_api, creds):
    route = jira_api.get("/myself")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"accountId": "a", "displayName": "Bot"}),
    ]
    from flyteplugins.jira import Config

    async with JiraClient(Config(retry_backoff=0.0)) as client:
        await client.get_myself()
    assert route.call_count == 2
