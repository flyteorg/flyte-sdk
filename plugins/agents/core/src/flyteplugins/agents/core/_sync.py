"""Sync bridge for adapter entry points whose implementation is async.

Every adapter's run_agent is an async function: the machinery underneath it
(model turns via flyte.trace, tool calls as child actions via task.aio, memory
and report I/O) is async end to end. A synchronous variant therefore has to run
that coroutine on some event loop. Two loops are deliberately avoided:

- The caller's thread's loop. Inside a Flyte sync task the body executes via
  run_sync_with_loop, so the current thread already has a running loop and
  asyncio.run() would raise. This also breaks agent SDKs' own sync wrappers
  (for example pydantic_ai run_sync), which is why adapters need this bridge.
- The SDK-global syncify loop. That loop is reserved for short control-plane
  I/O (see the guidance in flyte._trace); parking a whole agent run on it
  stalls every other syncify user in the process and risks deadlocks when
  tool code calls syncified APIs.

Instead the bridge delegates to flyte._utils.asyn.loop_manager, which keeps
one persistent background event loop per calling thread. That scoping gives
both properties at once: concurrent agents (necessarily on different threads,
since a sync call blocks its thread) get separate loops, so one agent's
blocking tool cannot stall another; and repeated calls from the same task
thread reuse a single live loop, so async resources the agent SDKs cache
across calls (HTTP clients, connection pools) stay bound to an open loop.
Context variables, including the Flyte task context, propagate through
run_coroutine_threadsafe's context capture.
"""

from __future__ import annotations

import functools
import typing

from flyte._utils.asyn import loop_manager

R = typing.TypeVar("R")


def run_coro_sync(coro: typing.Coroutine[typing.Any, typing.Any, R]) -> R:
    """Run an async-adapter coroutine to completion from synchronous code.

    Runs the coroutine on the calling thread's persistent background event
    loop (via flyte._utils.asyn.loop_manager). Exceptions propagate to the
    caller unchanged.
    """
    return loop_manager.run_sync(lambda: coro)


def sync_variant(afunc: typing.Callable[..., typing.Coroutine[typing.Any, typing.Any, R]]) -> typing.Callable[..., R]:
    """Build the synchronous companion of an async adapter entry point.

    Adapters use this to derive run_agent_sync from run_agent::

        run_agent_sync = sync_variant(run_agent)

    The wrapper keeps run_agent's signature and docstring for introspection and
    dispatches through :func:`run_coro_sync`.
    """

    @functools.wraps(afunc)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> R:
        return run_coro_sync(afunc(*args, **kwargs))

    wrapper.__name__ = f"{afunc.__name__}_sync"
    wrapper.__qualname__ = f"{afunc.__qualname__}_sync"
    doc = afunc.__doc__ or ""
    wrapper.__doc__ = (
        f"Synchronous variant of {afunc.__name__} for use in sync tasks; "
        f"runs the async implementation on a dedicated event loop.\n\n{doc}"
    )
    return wrapper
