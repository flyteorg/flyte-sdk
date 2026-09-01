"""Async Jira Cloud REST API client used by tasks, webhooks, and the MCP server.

Jira Cloud authenticates with HTTP Basic auth over an account email plus an API
token created at `id.atlassian.net`. Issue descriptions and comments use the
Atlassian Document Format (ADF); this client converts plain strings into the
minimal ADF paragraph shape so tasks can work in plain text.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx
from flyte.syncify import syncify

from ._config import Config, default_config
from ._errors import JiraAPIError, MissingCredentialsError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {500, 502, 503, 504}

#: Never sleep longer than this on a rate-limit retry. A reset window further out
#: is better surfaced as an error than silently held inside a task.
MAX_RATE_LIMIT_SLEEP = 60.0


def _retry_after_seconds(response: httpx.Response, fallback: float) -> float:
    """Seconds to wait from a `Retry-After` header, clamped and never raising.

    `Retry-After` is allowed to carry an HTTP-date instead of a delay in
    seconds; fall back to the caller's backoff rather than crashing on it.
    """
    raw = response.headers.get("Retry-After")
    try:
        delay = float(raw) if raw is not None else fallback
    except ValueError:
        delay = fallback
    return min(max(delay, 0.0), MAX_RATE_LIMIT_SLEEP)


def _text_to_adf(text: str) -> dict[str, Any]:
    """Convert a plain string into a minimal ADF document (one paragraph)."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def _adf_to_text(adf: Any) -> str:
    """Best-effort extraction of plain text from an ADF document."""
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return ""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                parts.append(node["text"])
            for child in node.get("content", []):
                walk(child)
            if node.get("type") in ("paragraph", "heading", "listItem"):
                parts.append("\n")
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(adf)
    return "".join(parts).strip()


def _simplify_issue(issue: dict[str, Any], base_url: str = "") -> dict[str, Any]:
    fields = issue.get("fields", {})
    return {
        "key": issue.get("key"),
        "id": issue.get("id"),
        "summary": fields.get("summary"),
        "status": (fields.get("status") or {}).get("name"),
        "issue_type": (fields.get("issuetype") or {}).get("name"),
        "assignee": (fields.get("assignee") or {}).get("displayName"),
        "reporter": (fields.get("reporter") or {}).get("displayName"),
        "priority": (fields.get("priority") or {}).get("name"),
        "description": _adf_to_text(fields.get("description")),
        "labels": fields.get("labels", []),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "url": f"{base_url}/browse/{issue.get('key')}" if base_url and issue.get("key") else None,
    }


class JiraClient:
    """Async client for the Jira Cloud REST API v3.

    Every method has two call forms. Use the async one on an event loop — in
    `async def` tasks, app handlers, and MCP tools:

    ```python
    from flyteplugins.jira import JiraClient

    async with JiraClient() as client:
        issue = await client.get_issue.aio("PROJ-123")
    ```

    Use the blocking one in plain `def` tasks and scripts. It parks the calling
    thread until the call returns, so never reach for it on an event loop:

    ```python
    with JiraClient() as client:
        issue = client.get_issue("PROJ-123")
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        base_url: Explicit Jira site URL. When omitted, read from the
            environment variable named by `config.base_url_env`.
        email: Explicit account email. When omitted, read from the
            environment variable named by `config.email_env`.
        api_token: Explicit API token. When omitted, read from the
            environment variable named by `config.api_token_env`.
    """

    def __init__(
        self,
        config: Config | None = None,
        base_url: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
    ):
        self.config = config or default_config
        self._base_url = base_url
        self._email = email
        self._api_token = api_token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> JiraClient:
        base_url = self._base_url if self._base_url is not None else self.config.base_url()
        if not base_url:
            raise MissingCredentialsError(self.config.base_url_env)
        email = self._email if self._email is not None else self.config.email()
        if not email:
            raise MissingCredentialsError(self.config.email_env)
        api_token = self._api_token if self._api_token is not None else self.config.api_token()
        if not api_token:
            raise MissingCredentialsError(self.config.api_token_env)

        self.base_url = base_url.rstrip("/")
        credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}{self.config.api_path}",
            headers={
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
            },
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def __enter__(self) -> JiraClient:
        """Enter synchronously, for use with the blocking call form.

        `__aenter__` runs on syncify's background loop — the same loop the
        syncified methods run on — so the underlying `httpx.AsyncClient` is
        created and used on a single loop.
        """
        return self._enter_sync()

    def __exit__(self, *exc_info: object) -> None:
        self._exit_sync()

    @syncify
    async def _enter_sync(self) -> JiraClient:
        return await self.__aenter__()

    @syncify
    async def _exit_sync(self) -> None:
        await self.__aexit__()

    @syncify
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send a request, retrying transient failures and 429s."""
        if self._client is None:
            raise RuntimeError("JiraClient must be used as an async context manager (async with ...).")

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, params=params, json=json)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise JiraAPIError(0, f"transport error: {exc}", url=path) from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code == 429:
                if attempt >= self.config.max_retries:
                    raise JiraAPIError(429, "rate limited", url=path)
                retry_after = _retry_after_seconds(response, backoff)
                logger.warning("Jira rate limited, retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code >= 400:
                raise JiraAPIError(
                    response.status_code, _error_message(response), url=str(response.url), body=_safe_json(response)
                )

            if not response.content:
                return None
            return response.json()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    @syncify
    async def get_myself(self) -> dict[str, Any]:
        """Return the authenticated user (`GET /myself`)."""
        data = await self.request.aio("GET", "/myself")
        return {
            "account_id": data.get("accountId"),
            "display_name": data.get("displayName"),
            "email": data.get("emailAddress"),
        }

    @syncify
    async def list_projects(self) -> list[dict[str, Any]]:
        """List projects visible to the authenticated user."""
        data = await self.request.aio("GET", "/project/search", params={"maxResults": 50})
        return [{"key": p.get("key"), "name": p.get("name"), "id": p.get("id")} for p in data.get("values", [])]

    @syncify
    async def get_issue(self, issue_key: str) -> dict[str, Any]:
        """Return a single issue by key (e.g. `PROJ-123`)."""
        data = await self.request.aio("GET", f"/issue/{issue_key}")
        return _simplify_issue(data, self.base_url)

    @syncify
    async def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search issues with JQL."""
        data = await self.request.aio("GET", "/search", params={"jql": jql, "maxResults": max_results})
        return [_simplify_issue(issue, self.base_url) for issue in data.get("issues", [])]

    @syncify
    async def list_comments(self, issue_key: str) -> list[dict[str, Any]]:
        """List comments on an issue."""
        data = await self.request.aio("GET", f"/issue/{issue_key}/comment")
        return [
            {
                "id": c.get("id"),
                "author": (c.get("author") or {}).get("displayName"),
                "body": _adf_to_text(c.get("body")),
                "created": c.get("created"),
            }
            for c in data.get("comments", [])
        ]

    @syncify
    async def list_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        """List the transitions available for an issue."""
        data = await self.request.aio("GET", f"/issue/{issue_key}/transitions")
        return [
            {"id": t.get("id"), "name": t.get("name"), "to_status": (t.get("to") or {}).get("name")}
            for t in data.get("transitions", [])
        ]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    @syncify
    async def create_issue(
        self,
        project_key: str,
        summary: str,
        issue_type: str = "Task",
        description: str | None = None,
        priority: str | None = None,
        labels: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an issue in a project.

        Args:
            project_key: The project key, e.g. `PROJ`.
            summary: One-line issue summary.
            issue_type: Issue type name (`Task`, `Bug`, `Story`, ...).
            description: Plain-text description, converted to ADF.
            priority: Optional priority name (`High`, `Medium`, ...).
            labels: Optional labels.
            extra_fields: Optional extra `fields` entries passed through
                verbatim for custom fields.
        """
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description is not None:
            fields["description"] = _text_to_adf(description)
        if priority is not None:
            fields["priority"] = {"name": priority}
        if labels:
            fields["labels"] = labels
        if extra_fields:
            fields.update(extra_fields)
        data = await self.request.aio("POST", "/issue", json={"fields": fields})
        return {"key": data.get("key"), "id": data.get("id"), "url": f"{self.base_url}/browse/{data.get('key')}"}

    @syncify
    async def update_issue(
        self,
        issue_key: str,
        summary: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an issue's summary, description, or labels."""
        fields: dict[str, Any] = {}
        if summary is not None:
            fields["summary"] = summary
        if description is not None:
            fields["description"] = _text_to_adf(description)
        if labels is not None:
            fields["labels"] = labels
        if extra_fields:
            fields.update(extra_fields)
        await self.request.aio("PUT", f"/issue/{issue_key}", json={"fields": fields})
        return {"key": issue_key}

    @syncify
    async def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        """Add a comment to an issue."""
        data = await self.request.aio("POST", f"/issue/{issue_key}/comment", json={"body": _text_to_adf(body)})
        return {"id": data.get("id"), "created": data.get("created")}

    @syncify
    async def transition_issue(self, issue_key: str, transition: str) -> dict[str, Any]:
        """Transition an issue by transition name or id.

        Looks up available transitions when a name is given; raises
        `JiraAPIError` when the name does not match.
        """
        if not transition.isdigit():
            transitions = await self.list_transitions.aio(issue_key)
            match = next((t for t in transitions if t["name"].lower() == transition.lower()), None)
            if match is None:
                available = ", ".join(t["name"] for t in transitions) or "<none>"
                raise JiraAPIError(
                    400, f"transition {transition!r} not available for {issue_key}; available: {available}"
                )
            transition_id = match["id"]
        else:
            transition_id = transition
        await self.request.aio("POST", f"/issue/{issue_key}/transitions", json={"transition": {"id": transition_id}})
        return {"key": issue_key, "transition": transition_id}

    @syncify
    async def delete_issue(self, issue_key: str) -> None:
        """Delete an issue permanently. Destructive and irreversible."""
        await self.request.aio("DELETE", f"/issue/{issue_key}")


def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def _error_message(response: httpx.Response) -> str:
    body = _safe_json(response)
    if body:
        messages = body.get("errorMessages") or []
        if messages:
            return "; ".join(str(m) for m in messages)
        errors = body.get("errors") or {}
        if errors:
            return "; ".join(f"{k}: {v}" for k, v in errors.items())
        if isinstance(body.get("message"), str):
            return body["message"]
    return response.text[:300] or f"HTTP {response.status_code}"
