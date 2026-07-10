"""OpenAI Agents ``Session`` backed by a keyed Flyte ``MemoryStore``.

The OpenAI Agents SDK reads and writes conversation history through a ``Session``
(``get_items`` / ``add_items`` / ``pop_item`` / ``clear_session``).
:class:`FlyteSession` implements that protocol over a durable, cross-run
:class:`~flyte.ai.agents.memory.MemoryStore` (keyed by a ``memory_key``), so
passing ``session=FlyteSession(store)`` to ``Runner.run`` gives the agent memory
that survives across runs and workers — backed by object storage rather than the
SDK's default local SQLite (which doesn't persist on a distributed backend).
"""

from __future__ import annotations

import typing


class FlyteSession:
    """An ``agents`` ``Session`` whose items live in a keyed Flyte ``MemoryStore``.

    The store's ``messages`` transcript is the session item list; ``add_items``
    persists durably (an object-store upload) so the next run for the same key
    resumes the conversation.
    """

    def __init__(self, store: typing.Any) -> None:
        self._store = store

    async def get_items(self, limit: int | None = None) -> list[dict[str, typing.Any]]:
        items = list(self._store.messages)
        return items[-limit:] if limit else items

    async def add_items(self, items: list[dict[str, typing.Any]]) -> None:
        if not items:
            return
        self._store.extend(items)
        await self._store.save.aio()

    async def pop_item(self) -> dict[str, typing.Any] | None:
        if not self._store.messages:
            return None
        item = self._store.messages.pop()
        await self._store.save.aio()
        return item

    async def clear_session(self) -> None:
        self._store.messages.clear()
        await self._store.save.aio()
