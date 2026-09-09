"""Process-wide Monty worker pool for sandboxed execution.

Since pydantic-monty 0.0.19, code runs in pooled `monty` subprocess workers
rather than in-process: `AsyncMonty` owns the pool and `checkout()` lends a
worker to a session for one execution. Spawning a worker costs a few hundred
milliseconds, so the pool is created once per process and shared by every
sandboxed task and `orchestrate_local` call; a warm checkout costs well under
a millisecond.

The pool is not bound to the event loop that entered it — it works from any
loop or thread — which lets one pool serve the main loop, the syncify loop and
worker threads alike. Workers are reaped when the pool object is dropped or
the process exits, so it is never explicitly closed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Optional

from ._config import SandboxedConfig

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession, ResourceLimits

# Exact pin the SDK is developed and tested against. pydantic-monty is pre-1.0
# and changes its API between patch releases, so the default sandbox image
# installs this exact version rather than whatever PyPI has latest.
MONTY_REQUIREMENT = "pydantic-monty==0.0.22"

_pool: Optional["AsyncMonty"] = None
_pool_lock = threading.Lock()


def _lazy_import_monty() -> Any:
    """Import `pydantic_monty` on first use, with a helpful error when missing."""
    try:
        import pydantic_monty
    except ImportError:
        raise ImportError(
            "pydantic-monty is required for sandboxed tasks. "
            f"Install it with: pip install 'flyte[sandbox]' or pip install '{MONTY_REQUIREMENT}'"
        ) from None
    return pydantic_monty


def _create_pool() -> "AsyncMonty":
    """Create and enter the shared pool.

    Runs on a helper thread under `_pool_lock` so concurrent first callers —
    on the same loop or on different loops — cannot each spawn a pool.
    `AsyncMonty.__aenter__` needs a running loop but does no work with
    `min_processes=0` (workers are spawned lazily by `checkout`), so a
    throwaway `asyncio.run` is the cheapest way to satisfy it.
    """
    global _pool  # noqa: PLW0603
    with _pool_lock:
        if _pool is None:
            monty = _lazy_import_monty()

            async def enter() -> "AsyncMonty":
                pool = monty.AsyncMonty(min_processes=0)
                await pool.__aenter__()
                return pool

            _pool = asyncio.run(enter())
        return _pool


async def get_pool() -> "AsyncMonty":
    """Return the process-wide `AsyncMonty` pool, creating it on first use."""
    if _pool is not None:
        return _pool
    return await asyncio.to_thread(_create_pool)


def _limits(config: SandboxedConfig) -> "ResourceLimits":
    """Translate `SandboxedConfig` into Monty `ResourceLimits`.

    `max_duration_secs` only counts time spent executing inside the worker;
    time the host spends servicing external calls (tasks, traces) is excluded,
    so a long-running tool chain does not trip the sandbox timeout.
    """
    return {
        "max_duration_secs": config.timeout_ms / 1000,
        "max_memory": config.max_memory,
        "max_recursion_depth": config.max_stack_depth,
    }


async def checkout(config: Optional[SandboxedConfig] = None) -> "AsyncMontySession":
    """Lend a pooled worker as an `AsyncMontySession` configured from *config*.

    Use as `async with await checkout(cfg) as session:`; the worker returns to
    the pool when the block exits, even if a feed was abandoned mid-way
    (e.g. an external call raised).
    """
    pool = await get_pool()
    return pool.checkout(limits=_limits(config or SandboxedConfig()))
