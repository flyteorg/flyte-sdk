"""Cross-run Deep Agents memory — a thin bridge over Flyte's keyed ``MemoryStore``.

A deep agent is driven with a messages state (``graph.ainvoke({"messages":
[...]})``) and also carries a virtual filesystem (the ``files`` state its
built-in filesystem tools read and write). By default neither survives the run.
This module bridges both: it resolves a keyed :class:`MemoryStore`, loads the
prior conversation and files from path-addressed JSON slots, and writes them
back after the run — so a later run with the same ``memory_key`` continues the
conversation *and* sees the same virtual filesystem.

The transcript is stored as ``messages_to_dict(...)`` output (rebuilt with
``messages_from_dict``); the files state is stored as its plain
``{path: contents}`` dict. All operations are best-effort: any failure leaves
the run untouched (memory never breaks a run).
"""

from __future__ import annotations

import typing

from flyte._logging import logger
from flyteplugins.agents.core import resolve_memory as _resolve_memory

if typing.TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# Path-addressed memory slots inside the MemoryStore.
_HISTORY_PATH = "deepagents/history.json"
_FILES_PATH = "deepagents/files.json"


async def resolve_memory(memory_key: str | None) -> typing.Any | None:
    """Resolve a keyed MemoryStore for Deep Agents cross-run memory, or ``None``.

    Best-effort: returns ``None`` when ``memory_key`` is falsy or no durable
    store can be resolved, so memory never breaks a run.
    """
    if not memory_key:
        return None
    return await _resolve_memory(memory_key)


async def load_history(store: typing.Any) -> list["BaseMessage"]:
    """Load and deserialize the prior conversation from ``store``.

    Returns an empty list when there is no prior history or on any error.
    """
    if store is None:
        return []
    try:
        from langchain_core.messages import messages_from_dict

        raw = await store.read_json.aio(_HISTORY_PATH, [])
        return messages_from_dict(raw) if raw else []
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load Deep Agents memory; continuing without prior history.")
        return []


async def load_files(store: typing.Any) -> dict[str, typing.Any]:
    """Load the agent's prior virtual filesystem (``{path: contents}``) from ``store``.

    Returns an empty dict when there are no prior files or on any error.
    """
    if store is None:
        return {}
    try:
        return dict(await store.read_json.aio(_FILES_PATH, {}) or {})
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not load Deep Agents files; continuing without prior files.")
        return {}


async def save_state(
    store: typing.Any,
    messages: typing.Sequence["BaseMessage"],
    files: typing.Mapping[str, typing.Any] | None = None,
) -> None:
    """Persist the conversation transcript and virtual filesystem to ``store``.

    Best-effort: logs and returns on any error so memory never breaks a run.
    """
    if store is None or not messages:
        return
    try:
        from langchain_core.messages import messages_to_dict

        await store.write_json.aio(_HISTORY_PATH, messages_to_dict(list(messages)))
        if files:
            await store.write_json.aio(_FILES_PATH, dict(files))
        await store.save.aio()
    except Exception:  # pragma: no cover - memory is best-effort, never fatal
        logger.warning("Could not save Deep Agents memory; conversation will not be resumed.")
