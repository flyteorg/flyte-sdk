"""Cross-run memory for Google ADK — persist and restore the session transcript.

ADK keeps the conversation as a list of ``Event``s on the session. For cross-run
memory we persist those events to a keyed ``MemoryStore`` and replay them into a
fresh session on the next run, so the agent continues the conversation. Keyed by a
stable ``memory_key`` (a user/thread id); best-effort and never fatal.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import resolve_memory

# Path-addressed slot holding the thread's ADK event transcript in the MemoryStore.
_EVENTS_PATH = "google/events.json"


async def load_memory(memory_key: str | None) -> tuple[typing.Any, list[typing.Any]]:
    """Return ``(store, prior_events)``; ``store`` is ``None`` when memory is off/unavailable."""
    from google.adk.events import Event

    store = await resolve_memory(memory_key)
    if store is None:
        return None, []

    raw = await store.read_json.aio(_EVENTS_PATH)
    events = [Event.model_validate(e) for e in (raw or [])]
    return store, events


async def save_memory(store: typing.Any, events: typing.Sequence[typing.Any]) -> None:
    """Persist the session's events to the keyed store (no-op when ``store`` is ``None``)."""
    if store is None:
        return

    payload = [e.model_dump(mode="json", exclude_none=True) for e in events]
    await store.write_json.aio(_EVENTS_PATH, payload, actor="google-agent")
    await store.save.aio()
