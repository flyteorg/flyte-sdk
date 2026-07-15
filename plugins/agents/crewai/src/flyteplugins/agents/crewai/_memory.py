"""Cross-run CrewAI memory — a thin handle over Flyte's keyed ``MemoryStore``.

CrewAI keeps conversation state in-memory and ``kickoff_async`` does not thread a
prior transcript back out (its ``LiteAgentOutput.messages`` holds only the
system+user turns of the current call, not the assistant reply). So this module
maintains the transcript itself: it resolves a keyed ``MemoryStore`` and stores
the running conversation — a list of ``{"role", "content"}`` dicts — in a
path-addressed JSON slot. A later run with the same ``memory_key`` loads that
transcript and prepends it to the new prompt, continuing the conversation.

Everything here is best-effort: a store that can't be resolved, or a read/write
that fails, degrades to a memoryless run rather than breaking it.
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

# Path-addressed memory slot holding a thread's conversation transcript
# (a list of ``{"role", "content"}`` message dicts) inside the MemoryStore.
_HISTORY_PATH = "crewai/history.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for CrewAI cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_history(store: typing.Any) -> list[dict[str, typing.Any]]:
    """Load the stored conversation transcript, or ``[]`` if none/unavailable."""
    if store is None:
        return []
    try:
        history = await store.read_json.aio(_HISTORY_PATH, default=[])
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load CrewAI conversation history; continuing without it.")
        return []
    return history if isinstance(history, list) else []


def build_input(history: list[dict[str, typing.Any]], user_input: str) -> list[dict[str, typing.Any]]:
    """Build the kickoff message list from prior transcript + the new user turn."""
    return [*history, {"role": "user", "content": user_input}]


async def save_turn(store: typing.Any, user_input: str, assistant_output: str) -> None:
    """Append the user + assistant turns to the transcript and persist it.

    Best-effort: any failure is logged and swallowed so a run never breaks on
    memory persistence.
    """
    if store is None:
        return
    try:
        history = await load_history(store)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": assistant_output})
        await store.write_json.aio(_HISTORY_PATH, history)
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not persist CrewAI conversation history; continuing.")
