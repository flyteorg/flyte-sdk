"""Durable, replayable steps via ``flyte.trace`` — the shared mechanism.

A model turn (or any expensive, replay-worthy call) is recorded as a
``flyte.trace`` leaf so that, inside a Flyte task, a crash/retry replays the
recorded result instead of re-running it. The real call is usually
non-serializable (closures, SDK objects), but ``flyte.trace`` keys its memo on
the decorated function's serializable arguments. :func:`durable_step` resolves
that by capturing the real call in a closure and feeding the trace only a
deterministic ``request_key``.
"""

from __future__ import annotations

import typing

import flyte

T = typing.TypeVar("T")


async def durable_step(
    request_key: str,
    run: typing.Callable[[], typing.Awaitable[typing.Any]],
    *,
    name: str = "durable_step",
    dumps: typing.Callable[[typing.Any], str] = lambda v: v,
    loads: typing.Callable[[str], typing.Any] = lambda v: v,
) -> typing.Any:
    """Run ``run()`` once as a durable, replayable trace step keyed by ``request_key``.

    The real (possibly non-serializable) work is captured in the ``run`` closure,
    so the traced function only ever sees the serializable ``request_key``. The
    result is serialized with ``dumps`` for the trace record (a ``str`` is stored
    inline and is human-readable in the Flyte UI) and rebuilt with ``loads`` on
    the way out and on replay.

    Args:
        request_key: A deterministic fingerprint of the call; the trace memo key.
        run: Zero-arg async callable performing the real work.
        name: Label for the trace action in the Flyte UI.
        dumps: Serialize the result to a ``str`` for the trace record.
        loads: Rebuild the result from the recorded ``str``.

    Outside a task context ``flyte.trace`` is a transparent pass-through, so this
    also works unchanged for local runs and unit tests.
    """

    async def _step(key: str) -> str:
        return dumps(await run())

    # Name the closure so the trace action reads meaningfully in the UI.
    _step.__name__ = name
    _step.__qualname__ = name
    return loads(await flyte.trace(_step)(request_key))
