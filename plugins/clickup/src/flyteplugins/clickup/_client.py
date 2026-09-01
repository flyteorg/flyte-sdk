"""Async ClickUp REST API client used by tasks, webhooks, and the MCP server.

The client wraps ClickUp's REST API v2 with retry on transient failures and
429 rate limits, and exposes one method per operation. It deliberately
includes `list_statuses` so workflows can validate a status before attempting
an update — ClickUp rejects transitions to statuses a list does not have, and
pre-checking produces a far better error than a blind 400.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from flyte.syncify import syncify

from ._config import Config, default_config
from ._errors import ClickUpAPIError, MissingCredentialsError

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


def _simplify_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "description": task.get("description") or "",
        "status": (task.get("status") or {}).get("status"),
        "priority": (task.get("priority") or {}).get("priority"),
        "url": task.get("url"),
        "list_id": (task.get("list") or {}).get("id"),
        "assignees": [a.get("username") for a in task.get("assignees", [])],
        "tags": [t.get("name") for t in task.get("tags", [])],
        "created_at": task.get("date_created"),
        "updated_at": task.get("date_updated"),
    }


class ClickUpClient:
    """Async client for the ClickUp REST API v2.

    Use as an async context manager:

    ```python
    from flyteplugins.clickup import ClickUpClient

    async with ClickUpClient() as client:
        task = await client.get_task("1a2b3c")
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        token: Explicit personal API token. When omitted, the token is read
            from the environment variable named by `config.token_env`.
    """

    def __init__(self, config: Config | None = None, token: str | None = None):
        self.config = config or default_config
        self._token = token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ClickUpClient:
        token = self._token if self._token is not None else self.config.token()
        if not token:
            raise MissingCredentialsError(self.config.token_env)
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            headers={
                "Authorization": token,
                "ClickUp-Client": self.config.client_id,
            },
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def __enter__(self) -> ClickUpClient:
        """Enter synchronously, for use with the blocking call form.

        `__aenter__` runs on syncify's background loop — the same loop the
        syncified methods run on — so the underlying `httpx.AsyncClient` is
        created and used on a single loop.
        """
        return self._enter_sync()

    def __exit__(self, *exc_info: object) -> None:
        self._exit_sync()

    @syncify
    async def _enter_sync(self) -> ClickUpClient:
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
            raise RuntimeError("ClickUpClient must be used as an async context manager (async with ...).")

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, params=params, json=json)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise ClickUpAPIError(0, f"transport error: {exc}", url=path) from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code == 429:
                if attempt >= self.config.max_retries:
                    raise ClickUpAPIError(429, "rate limited", url=path)
                retry_after = _retry_after_seconds(response, backoff)
                logger.warning("ClickUp rate limited, retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code >= 400:
                raise ClickUpAPIError(
                    response.status_code, _error_message(response), url=str(response.url), body=_safe_json(response)
                )

            if not response.content:
                return None
            return response.json()

    # ------------------------------------------------------------------
    # reads: workspace structure
    # ------------------------------------------------------------------

    @syncify
    async def get_user(self) -> dict[str, Any]:
        """Return the authenticated user."""
        data = await self.request.aio("GET", "/user")
        user = data.get("user", {})
        return {"id": user.get("id"), "username": user.get("username"), "email": user.get("email")}

    @syncify
    async def list_workspaces(self) -> list[dict[str, Any]]:
        """List the workspaces (teams) the token can access."""
        data = await self.request.aio("GET", "/team")
        return [{"id": t.get("id"), "name": t.get("name"), "color": t.get("color")} for t in data.get("teams", [])]

    @syncify
    async def list_spaces(self, workspace_id: str) -> list[dict[str, Any]]:
        """List spaces in a workspace."""
        data = await self.request.aio("GET", f"/team/{workspace_id}/space")
        return [{"id": s.get("id"), "name": s.get("name")} for s in data.get("spaces", [])]

    @syncify
    async def list_folders(self, space_id: str) -> list[dict[str, Any]]:
        """List folders in a space."""
        data = await self.request.aio("GET", f"/space/{space_id}/folder")
        return [{"id": f.get("id"), "name": f.get("name")} for f in data.get("folders", [])]

    @syncify
    async def list_lists(self, space_id: str | None = None, folder_id: str | None = None) -> list[dict[str, Any]]:
        """List task lists in a space (including folderless lists) or folder."""
        if folder_id:
            data = await self.request.aio("GET", f"/folder/{folder_id}/list")
        elif space_id:
            data = await self.request.aio("GET", f"/space/{space_id}/list")
        else:
            raise ValueError("pass either space_id or folder_id")
        return [{"id": item.get("id"), "name": item.get("name")} for item in data.get("lists", [])]

    @syncify
    async def list_statuses(self, list_id: str) -> list[str]:
        """List the valid status names of a task list, in workflow order.

        Use this before `update_task(..., status=...)`: ClickUp rejects
        transitions to statuses the list does not define.
        """
        data = await self.request.aio("GET", f"/list/{list_id}")
        return [status.get("status") for status in data.get("statuses", []) if status.get("status")]

    # ------------------------------------------------------------------
    # reads: tasks and comments
    # ------------------------------------------------------------------

    @syncify
    async def list_tasks(
        self, list_id: str, statuses: list[str] | None = None, archived: bool = False
    ) -> list[dict[str, Any]]:
        """List tasks in a task list, optionally filtered by status."""
        params: dict[str, Any] = {"archived": str(archived).lower()}
        if statuses:
            params["statuses[]"] = statuses
        data = await self.request.aio("GET", f"/list/{list_id}/task", params=params)
        return [_simplify_task(t) for t in data.get("tasks", [])]

    @syncify
    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Return a single task."""
        data = await self.request.aio("GET", f"/task/{task_id}")
        return _simplify_task(data)

    @syncify
    async def list_comments(self, task_id: str) -> list[dict[str, Any]]:
        """List comments on a task."""
        data = await self.request.aio("GET", f"/task/{task_id}/comment")
        return [
            {
                "id": c.get("id"),
                "text": c.get("comment", [{}])[0].get("text", "") if c.get("comment") else "",
                "user": (c.get("user") or {}).get("username"),
                "date": c.get("date"),
            }
            for c in data.get("comments", [])
        ]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    @syncify
    async def create_task(
        self,
        list_id: str,
        name: str,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        assignee_ids: list[int] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a task in a list.

        Priority: 1 (urgent), 2 (high), 3 (normal), 4 (low). Validate `status`
        against `list_statuses` first.
        """
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if status is not None:
            payload["status"] = status
        if priority is not None:
            payload["priority"] = priority
        if assignee_ids:
            payload["assignees"] = assignee_ids
        if tags:
            payload["tags"] = tags
        task = await self.request.aio("POST", f"/list/{list_id}/task", json=payload)
        return _simplify_task(task)

    @syncify
    async def update_task(
        self,
        task_id: str,
        name: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        assignee_ids: list[int] | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a task. Pass only the fields to change.

        Validate `status` against `list_statuses` first — ClickUp rejects
        transitions to statuses the task's list does not define.
        """
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if status is not None:
            payload["status"] = status
        if priority is not None:
            payload["priority"] = priority
        if assignee_ids is not None:
            payload["assignees"] = assignee_ids
        if add_tags:
            payload["add_tags"] = add_tags
        if remove_tags:
            payload["remove_tags"] = remove_tags
        task = await self.request.aio("PUT", f"/task/{task_id}", json=payload)
        return _simplify_task(task)

    @syncify
    async def add_comment(self, task_id: str, text: str) -> dict[str, Any]:
        """Comment on a task."""
        data = await self.request.aio("POST", f"/task/{task_id}/comment", json={"comment_text": text})
        return {"id": data.get("id")}

    @syncify
    async def delete_task(self, task_id: str) -> None:
        """Delete a task permanently. Destructive and irreversible."""
        await self.request.aio("DELETE", f"/task/{task_id}")


def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def _error_message(response: httpx.Response) -> str:
    body = _safe_json(response)
    if body:
        for key in ("err", "error", "message"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return response.text[:300] or f"HTTP {response.status_code}"
