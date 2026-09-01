"""Async Notion API client used by tasks, the polling app, and the MCP server.

Notion has no webhooks, so "reacting to events" is done by polling: `NotionClient`
exposes `query_database` and `search` with filters, and the app environment's
`/api/poll` endpoint wraps them for scheduled or manual triggering.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from flyte.syncify import syncify

from ._config import Config, default_config
from ._errors import MissingCredentialsError, NotionAPIError
from ._helpers import extract_title, title_property

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


def _simplify_page(page: dict[str, Any]) -> dict[str, Any]:
    parent = page.get("parent") or {}
    return {
        "id": page.get("id"),
        "url": page.get("url"),
        "title": extract_title(page.get("properties") or {}),
        "archived": page.get("archived", False),
        "created_time": page.get("created_time"),
        "last_edited_time": page.get("last_edited_time"),
        "parent_id": parent.get("database_id") or parent.get("page_id") or parent.get("workspace"),
        "parent_type": parent.get("type"),
    }


def _simplify_database(db: dict[str, Any]) -> dict[str, Any]:
    title = "".join(t.get("plain_text", "") for t in db.get("title", []))
    return {
        "id": db.get("id"),
        "title": title,
        "url": db.get("url"),
        "description": "".join(t.get("plain_text", "") for t in db.get("description", [])),
        "properties": {name: {"type": prop.get("type")} for name, prop in (db.get("properties") or {}).items()},
    }


class NotionClient:
    """Async client for the Notion API.

    Use as an async context manager:

    ```python
    from flyteplugins.notion import NotionClient

    async with NotionClient() as client:
        me = await client.get_me()
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        token: Explicit integration token. When omitted, the token is read
            from the environment variable named by `config.token_env`.
    """

    def __init__(self, config: Config | None = None, token: str | None = None):
        self.config = config or default_config
        self._token = token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> NotionClient:
        token = self._token if self._token is not None else self.config.token()
        if not token:
            raise MissingCredentialsError(self.config.token_env)
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": self.config.notion_version,
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def __enter__(self) -> NotionClient:
        """Enter synchronously, for use with the blocking call form.

        `__aenter__` runs on syncify's background loop — the same loop the
        syncified methods run on — so the underlying `httpx.AsyncClient` is
        created and used on a single loop.
        """
        return self._enter_sync()

    def __exit__(self, *exc_info: object) -> None:
        self._exit_sync()

    @syncify
    async def _enter_sync(self) -> NotionClient:
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
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request, retrying transient failures and 429s."""
        if self._client is None:
            raise RuntimeError("NotionClient must be used as an async context manager (async with ...).")

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, json=json, params=params)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise NotionAPIError(0, f"transport error: {exc}", url=path) from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code == 429:
                if attempt >= self.config.max_retries:
                    raise NotionAPIError(429, "rate limited", code="rate_limited", url=path)
                retry_after = _retry_after_seconds(response, backoff)
                logger.warning("Notion rate limited, retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code >= 400:
                body = _safe_json(response) or {}
                raise NotionAPIError(
                    response.status_code,
                    body.get("message", response.text[:300] or f"HTTP {response.status_code}"),
                    code=body.get("code", ""),
                    url=str(response.url),
                    body=body,
                )

            return response.json()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    @syncify
    async def get_me(self) -> dict[str, Any]:
        """Return the integration's own user (bot) object."""
        data = await self.request.aio("GET", "/users/me")
        return {"id": data.get("id"), "name": data.get("name"), "type": data.get("type")}

    @syncify
    async def search(
        self,
        query: str = "",
        object_type: str | None = None,
        page_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Search pages and databases visible to the integration.

        Args:
            query: Text to match against titles.
            object_type: Optionally restrict to `page` or `database`.
            page_size: Maximum results.
        """
        payload: dict[str, Any] = {"query": query, "page_size": page_size}
        if object_type:
            payload["filter"] = {"property": "object", "value": object_type}
        data = await self.request.aio("POST", "/search", json=payload)
        return [_simplify_result(result) for result in data.get("results", [])]

    @syncify
    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Return a page's metadata and simplified title."""
        data = await self.request.aio("GET", f"/pages/{page_id}")
        return _simplify_page(data)

    @syncify
    async def get_database(self, database_id: str) -> dict[str, Any]:
        """Return a database's title, description, and property schema."""
        data = await self.request.aio("GET", f"/databases/{database_id}")
        return _simplify_database(data)

    @syncify
    async def query_database(
        self,
        database_id: str,
        filter: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None,
        page_size: int = 50,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Query a database and return simplified pages plus pagination info.

        Returns:
            `{"pages": [...], "next_cursor", "has_more"}`.
        """
        payload: dict[str, Any] = {"page_size": page_size}
        if filter:
            payload["filter"] = filter
        if sorts:
            payload["sorts"] = sorts
        if start_cursor:
            payload["start_cursor"] = start_cursor
        data = await self.request.aio("POST", f"/databases/{database_id}/query", json=payload)
        return {
            "pages": [_simplify_page(page) for page in data.get("results", [])],
            "next_cursor": data.get("next_cursor"),
            "has_more": data.get("has_more", False),
        }

    @syncify
    async def query_database_since(
        self, database_id: str, last_edited_after: str, page_size: int = 100
    ) -> list[dict[str, Any]]:
        """Return pages in a database edited after an ISO 8601 timestamp.

        This is the polling primitive for reacting to Notion changes: call it
        periodically with the previous poll time and act on the results.
        """
        result = await self.query_database.aio(
            database_id,
            filter={"timestamp": "last_edited_time", "last_edited_time": {"after": last_edited_after}},
            sorts=[{"timestamp": "last_edited_time", "direction": "ascending"}],
            page_size=page_size,
        )
        return result["pages"]

    @syncify
    async def list_block_children(self, block_id: str, page_size: int = 100) -> list[dict[str, Any]]:
        """Return the child blocks of a page or block."""
        data = await self.request.aio("GET", f"/blocks/{block_id}/children", params={"page_size": page_size})
        return data.get("results", [])

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    @syncify
    async def create_database_page(self, database_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        """Create a page (row) in a database.

        Build property values with the helpers in `flyteplugins.notion`
        (`title_property`, `select_property`, ...).
        """
        data = await self.request.aio(
            "POST", "/pages", json={"parent": {"database_id": database_id}, "properties": properties}
        )
        return _simplify_page(data)

    @syncify
    async def create_page(
        self,
        parent_page_id: str,
        title: str = "Untitled",
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a child page under another page, optionally with content blocks."""
        payload: dict[str, Any] = {
            "parent": {"page_id": parent_page_id},
            "properties": {"title": title_property(title)},
        }
        if blocks:
            payload["children"] = blocks
        data = await self.request.aio("POST", "/pages", json=payload)
        return _simplify_page(data)

    @syncify
    async def update_page(
        self,
        page_id: str,
        properties: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        """Update a page's properties and/or archived state."""
        payload: dict[str, Any] = {}
        if properties:
            payload["properties"] = properties
        if archived is not None:
            payload["archived"] = archived
        data = await self.request.aio("PATCH", f"/pages/{page_id}", json=payload)
        return _simplify_page(data)

    @syncify
    async def append_blocks(self, block_id: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append content blocks to a page or block (up to 100 per call)."""
        data = await self.request.aio("PATCH", f"/blocks/{block_id}/children", json={"children": blocks})
        return data.get("results", [])

    @syncify
    async def archive_page(self, page_id: str) -> dict[str, Any]:
        """Archive (soft-delete) a page. Destructive: removes it from views."""
        return await self.update_page.aio(page_id, archived=True)


def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def _simplify_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("object") == "database":
        return {"object": "database", **_simplify_database(result)}
    return {"object": "page", **_simplify_page(result)}
