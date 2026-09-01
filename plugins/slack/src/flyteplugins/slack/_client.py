"""Async Slack Web API client used by tasks, event handlers, and the MCP server.

The Slack Web API is unusual: most methods return HTTP 200 even on failure,
signaling success with a top-level `ok` boolean and errors with an `error`
code. This client normalizes that convention into exceptions, retries
transient failures and 429 rate limits, and exposes one method per API call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from flyte.syncify import syncify

from ._config import Config, default_config
from ._errors import MissingCredentialsError, SlackAPIError

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


class SlackClient:
    """Async client for the Slack Web API.

    Use as an async context manager:

    ```python
    from flyteplugins.slack import SlackClient

    async with SlackClient() as client:
        message = await client.post_message("C12345", "hello from Flyte")
    ```

    Args:
        config: Plugin configuration. Defaults to the module-level
            `default_config`.
        bot_token: Explicit bot token (`xoxb-...`). When omitted, the token
            is read from the environment variable named by
            `config.bot_token_env`.
    """

    def __init__(self, config: Config | None = None, bot_token: str | None = None):
        self.config = config or default_config
        self._bot_token = bot_token
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SlackClient:
        token = self._bot_token if self._bot_token is not None else self.config.bot_token()
        if not token:
            raise MissingCredentialsError(self.config.bot_token_env)
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": self.config.user_agent,
            },
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def __enter__(self) -> SlackClient:
        """Enter synchronously, for use with the blocking call form.

        `__aenter__` runs on syncify's background loop — the same loop the
        syncified methods run on — so the underlying `httpx.AsyncClient` is
        created and used on a single loop.
        """
        return self._enter_sync()

    def __exit__(self, *exc_info: object) -> None:
        self._exit_sync()

    @syncify
    async def _enter_sync(self) -> SlackClient:
        return await self.__aenter__()

    @syncify
    async def _exit_sync(self) -> None:
        await self.__aexit__()

    @syncify
    async def request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request, retrying transient failures and 429s.

        Returns the parsed JSON body and raises `SlackAPIError` when Slack
        reports `ok: false` or returns a non-2xx status.
        """
        if self._client is None:
            raise RuntimeError("SlackClient must be used as an async context manager (async with ...).")

        backoff = self.config.retry_backoff
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, json=json)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise SlackAPIError(f"transport error: {exc}", status_code=0, url=path) from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code == 429:
                if attempt >= self.config.max_retries:
                    raise SlackAPIError("rate_limited", status_code=429, url=path)
                retry_after = _retry_after_seconds(response, backoff)
                logger.warning("Slack rate limited, retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1
                continue

            if response.status_code >= 400:
                raise SlackAPIError(f"HTTP {response.status_code}", status_code=response.status_code, url=path)

            data = response.json()
            if not data.get("ok", False):
                raise SlackAPIError(data.get("error", "unknown_error"), url=path)
            return data

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    @syncify
    async def post_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        unfurl_links: bool = False,
    ) -> dict[str, Any]:
        """Post a message to a channel, optionally as a thread reply.

        Returns `{"channel", "ts", "permalink"}`; `permalink` is best-effort
        and omitted when it cannot be fetched.
        """
        payload: dict[str, Any] = {"channel": channel, "text": text, "unfurl_links": unfurl_links}
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts
        data = await self.request.aio("POST", "/chat.postMessage", json=payload)
        result = {"channel": data.get("channel"), "ts": data.get("ts")}
        try:
            permalink = await self.get_message_permalink.aio(result["channel"], result["ts"])
            result["permalink"] = permalink
        except SlackAPIError:
            pass
        return result

    @syncify
    async def update_message(self, channel: str, ts: str, text: str) -> dict[str, Any]:
        """Update an existing message in place."""
        data = await self.request.aio("POST", "/chat.update", json={"channel": channel, "ts": ts, "text": text})
        return {"channel": data.get("channel"), "ts": data.get("ts")}

    @syncify
    async def reply_in_thread(self, channel: str, thread_ts: str, text: str) -> dict[str, Any]:
        """Post a reply in a thread rooted at `thread_ts`."""
        return await self.post_message.aio(channel, text, thread_ts=thread_ts)

    @syncify
    async def get_message_permalink(self, channel: str, message_ts: str) -> str:
        """Return the permanent link to a message."""
        data = await self.request.aio("GET", f"/chat.getPermalink?channel={channel}&message_ts={message_ts}")
        return data.get("permalink", "")

    # ------------------------------------------------------------------
    # channels and history
    # ------------------------------------------------------------------

    @syncify
    async def list_channels(self, types: str = "public_channel", limit: int = 100) -> list[dict[str, Any]]:
        """List conversations (channels) visible to the bot."""
        data = await self.request.aio("GET", f"/conversations.list?types={types}&limit={limit}")
        return [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "is_private": c.get("is_private", False),
                "is_member": c.get("is_member", False),
            }
            for c in data.get("channels", [])
        ]

    @syncify
    async def get_channel(self, channel: str) -> dict[str, Any]:
        """Return metadata for a channel by id."""
        data = await self.request.aio("GET", f"/conversations.info?channel={channel}")
        info = data.get("channel", {})
        return {
            "id": info.get("id"),
            "name": info.get("name"),
            "topic": (info.get("topic") or {}).get("value", ""),
            "purpose": (info.get("purpose") or {}).get("value", ""),
            "is_private": info.get("is_private", False),
        }

    @syncify
    async def get_channel_history(self, channel: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent messages in a channel (newest last)."""
        data = await self.request.aio("GET", f"/conversations.history?channel={channel}&limit={limit}")
        messages = data.get("messages", [])
        return [_simplify_message(m) for m in reversed(messages)]

    @syncify
    async def get_thread(self, channel: str, thread_ts: str) -> list[dict[str, Any]]:
        """Return all messages in a thread, oldest first."""
        data = await self.request.aio("GET", f"/conversations.replies?channel={channel}&ts={thread_ts}")
        return [_simplify_message(m) for m in data.get("messages", [])]

    # ------------------------------------------------------------------
    # users, reactions, channel management
    # ------------------------------------------------------------------

    @syncify
    async def get_user(self, user_id: str) -> dict[str, Any]:
        """Return profile information for a user."""
        data = await self.request.aio("GET", f"/users.info?user={user_id}")
        user = data.get("user", {})
        profile = user.get("profile", {})
        return {
            "id": user.get("id"),
            "name": user.get("name"),
            "real_name": profile.get("real_name") or user.get("real_name", ""),
            "display_name": profile.get("display_name", ""),
            "is_bot": user.get("is_bot", False),
        }

    @syncify
    async def add_reaction(self, channel: str, message_ts: str, emoji: str) -> bool:
        """Add an emoji reaction to a message (name without colons)."""
        await self.request.aio(
            "POST",
            "/reactions.add",
            json={"channel": channel, "timestamp": message_ts, "name": emoji.strip(":")},
        )
        return True

    @syncify
    async def remove_reaction(self, channel: str, message_ts: str, emoji: str) -> bool:
        """Remove an emoji reaction from a message."""
        await self.request.aio(
            "POST",
            "/reactions.remove",
            json={"channel": channel, "timestamp": message_ts, "name": emoji.strip(":")},
        )
        return True

    @syncify
    async def create_channel(self, name: str, is_private: bool = False) -> dict[str, Any]:
        """Create a channel and return its id and name."""
        data = await self.request.aio("POST", "/conversations.create", json={"name": name, "is_private": is_private})
        channel = data.get("channel", {})
        return {"id": channel.get("id"), "name": channel.get("name")}


def _simplify_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": message.get("ts"),
        "thread_ts": message.get("thread_ts"),
        "user": message.get("user"),
        "text": message.get("text", ""),
        "subtype": message.get("subtype"),
    }
