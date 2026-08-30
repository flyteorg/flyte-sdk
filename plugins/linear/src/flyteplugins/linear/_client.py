"""Async Linear GraphQL API client used by tasks, webhooks, and the MCP server.

Linear's public API is a single GraphQL endpoint. This client wraps it with
retry on transient failures and exposes one method per operation, returning
simplified dicts rather than raw GraphQL responses.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from ._config import Config, default_config
from ._errors import LinearAPIError, MissingCredentialsError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {500, 502, 503, 504}

_ISSUE_FIELDS = """
fragment IssueFields on Issue {
  id
  identifier
  title
  description
  url
  priority
  createdAt
  updatedAt
  state { id name type }
  assignee { id name displayName }
  team { id key name }
  labels { nodes { id name } }
}
"""


def _simplify_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue.get("id"),
        "identifier": issue.get("identifier"),
        "title": issue.get("title"),
        "description": issue.get("description") or "",
        "url": issue.get("url"),
        "priority": issue.get("priority"),
        "state": (issue.get("state") or {}).get("name"),
        "state_id": (issue.get("state") or {}).get("id"),
        "assignee": (issue.get("assignee") or {}).get("displayName") or (issue.get("assignee") or {}).get("name"),
        "team": (issue.get("team") or {}).get("key"),
        "labels": [label["name"] for label in (issue.get("labels") or {}).get("nodes", [])],
        "created_at": issue.get("createdAt"),
        "updated_at": issue.get("updatedAt"),
    }


class LinearClient:
    """Async client for the Linear GraphQL API.

    Use as an async context manager:

    ```python
    from flyteplugins.linear import LinearClient

    async with LinearClient() as client:
        issue = await client.get_issue("ENG-123")
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        api_key: Explicit API key. When omitted, the key is read from the
            environment variable named by `config.api_key_env`.
    """

    def __init__(self, config: Config | None = None, api_key: str | None = None):
        self.config = config or default_config
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LinearClient:
        key = self._api_key if self._api_key is not None else self.config.api_key()
        if not key:
            raise MissingCredentialsError(self.config.api_key_env)
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": key,
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a GraphQL query or mutation and return its `data` object.

        Raises `LinearAPIError` on HTTP failures or when the response carries
        GraphQL errors.
        """
        if self._client is None:
            raise RuntimeError("LinearClient must be used as an async context manager (async with ...).")

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    self.config.api_base_url, json={"query": query, "variables": variables or {}}
                )
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise LinearAPIError(f"transport error: {exc}", status_code=0) from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code >= 400:
                raise LinearAPIError(
                    f"HTTP {response.status_code}: {response.text[:300]}", status_code=response.status_code
                )

            body = response.json()
            errors = body.get("errors") or []
            if errors:
                raise LinearAPIError(errors[0].get("message", "unknown error"), errors=errors)
            return body.get("data") or {}

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    async def get_viewer(self) -> dict[str, Any]:
        """Return the authenticated user (`viewer`)."""
        data = await self.graphql("query { viewer { id name displayName email } }")
        viewer = data.get("viewer", {})
        return {
            "id": viewer.get("id"),
            "name": viewer.get("displayName") or viewer.get("name"),
            "email": viewer.get("email"),
        }

    async def list_teams(self) -> list[dict[str, Any]]:
        """List the workspace's teams."""
        data = await self.graphql("query { teams(first: 50) { nodes { id key name } } }")
        return data.get("teams", {}).get("nodes", [])

    async def list_workflow_states(self, team_id: str) -> list[dict[str, Any]]:
        """List workflow states (Backlog, In Progress, Done, ...) for a team."""
        data = await self.graphql(
            "query States($teamId: String!) { workflowStates(teamId: $teamId) { nodes { id name type position } } }",
            {"teamId": team_id},
        )
        states = data.get("workflowStates", {}).get("nodes", [])
        return sorted(states, key=lambda s: s.get("position") or 0)

    async def list_issues(
        self,
        team_key: str | None = None,
        state: str | None = None,
        assignee: str | None = None,
        first: int = 50,
    ) -> list[dict[str, Any]]:
        """List issues, optionally filtered by team, workflow state, or assignee."""
        conditions: list[dict[str, Any]] = []
        if team_key:
            conditions.append({"team": {"key": {"eq": team_key}}})
        if state:
            conditions.append({"state": {"name": {"eq": state}}})
        if assignee:
            conditions.append({"assignee": {"displayName": {"eq": assignee}}})
        variables: dict[str, Any] = {"first": first}
        if conditions:
            variables["filter"] = {"and": conditions}
        query = (
            "query ListIssues($first: Int!, $filter: IssueFilter) {"
            f" issues(first: $first, filter: $filter) {{ nodes {{ ...IssueFields }} }} }}\n{_ISSUE_FIELDS}"
        )
        data = await self.graphql(query, variables)
        return [_simplify_issue(issue) for issue in data.get("issues", {}).get("nodes", [])]

    async def get_issue(self, identifier: str) -> dict[str, Any]:
        """Fetch one issue by identifier (`TEAM-123`) or by UUID.

        Raises:
            LinearAPIError: when the issue cannot be found.
        """
        match = re.fullmatch(r"([A-Za-z0-9]+)-(\d+)", identifier)
        if match:
            team_key, number = match.group(1), int(match.group(2))
            data = await self.graphql(
                "query FindIssue($teamKey: String!, $number: Float!) {"
                " issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } })"
                f" {{ nodes {{ ...IssueFields }} }} }}\n{_ISSUE_FIELDS}",
                {"teamKey": team_key, "number": number},
            )
            nodes = data.get("issues", {}).get("nodes", [])
            if not nodes:
                raise LinearAPIError(f"issue {identifier} not found")
            return _simplify_issue(nodes[0])

        data = await self.graphql(
            f"query GetIssue($id: String!) {{ issue(id: $id) {{ ...IssueFields }} }}\n{_ISSUE_FIELDS}",
            {"id": identifier},
        )
        issue = data.get("issue")
        if not issue:
            raise LinearAPIError(f"issue {identifier} not found")
        return _simplify_issue(issue)

    async def list_comments(self, issue_id: str) -> list[dict[str, Any]]:
        """List comments on an issue (by issue UUID)."""
        data = await self.graphql(
            "query Comments($issueId: String!) { comments(filter: { issue: { id: { eq: $issueId } } })"
            " { nodes { id body createdAt user { displayName name } url } } }",
            {"issueId": issue_id},
        )
        return [
            {
                "id": c.get("id"),
                "body": c.get("body") or "",
                "user": (c.get("user") or {}).get("displayName") or (c.get("user") or {}).get("name"),
                "url": c.get("url"),
                "created_at": c.get("createdAt"),
            }
            for c in data.get("comments", {}).get("nodes", [])
        ]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    async def create_issue(
        self,
        team_id: str,
        title: str,
        description: str | None = None,
        priority: int | None = None,
        assignee_id: str | None = None,
        label_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an issue in a team (use `list_teams` to resolve the team id)."""
        issue_input: dict[str, Any] = {"teamId": team_id, "title": title}
        if description is not None:
            issue_input["description"] = description
        if priority is not None:
            issue_input["priority"] = priority
        if assignee_id:
            issue_input["assigneeId"] = assignee_id
        if label_ids:
            issue_input["labelIds"] = label_ids
        data = await self.graphql(
            "mutation CreateIssue($input: IssueCreateInput!) {"
            f" issueCreate(input: $input) {{ success issue {{ ...IssueFields }} }} }}\n{_ISSUE_FIELDS}",
            {"input": issue_input},
        )
        payload = data.get("issueCreate", {})
        if not payload.get("success"):
            raise LinearAPIError("issueCreate returned success=false")
        return _simplify_issue(payload.get("issue") or {})

    async def update_issue(
        self,
        issue_id: str,
        title: str | None = None,
        description: str | None = None,
        state_id: str | None = None,
        assignee_id: str | None = None,
        priority: int | None = None,
    ) -> dict[str, Any]:
        """Update an issue (by UUID). Pass only the fields to change.

        Use `list_workflow_states` to resolve a workflow state id.
        """
        issue_input: dict[str, Any] = {}
        if title is not None:
            issue_input["title"] = title
        if description is not None:
            issue_input["description"] = description
        if state_id is not None:
            issue_input["stateId"] = state_id
        if assignee_id is not None:
            issue_input["assigneeId"] = assignee_id
        if priority is not None:
            issue_input["priority"] = priority
        data = await self.graphql(
            "mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {"
            f" issueUpdate(id: $id, input: $input) {{ success issue {{ ...IssueFields }} }} }}\n{_ISSUE_FIELDS}",
            {"id": issue_id, "input": issue_input},
        )
        payload = data.get("issueUpdate", {})
        if not payload.get("success"):
            raise LinearAPIError("issueUpdate returned success=false")
        return _simplify_issue(payload.get("issue") or {})

    async def add_comment(self, issue_id: str, body: str) -> dict[str, Any]:
        """Add a comment to an issue (by UUID)."""
        data = await self.graphql(
            "mutation AddComment($input: CommentCreateInput!) {"
            " commentCreate(input: $input) { success comment { id body url createdAt } } }",
            {"input": {"issueId": issue_id, "body": body}},
        )
        payload = data.get("commentCreate", {})
        if not payload.get("success"):
            raise LinearAPIError("commentCreate returned success=false")
        comment = payload.get("comment") or {}
        return {"id": comment.get("id"), "url": comment.get("url"), "created_at": comment.get("createdAt")}
