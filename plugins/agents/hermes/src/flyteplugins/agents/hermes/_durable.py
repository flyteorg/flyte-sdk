"""Durable, replayable agent runs for Hermes.

The real adapters (see ``openai``/``pydantic_ai``) make each *model turn* durable
by wrapping the SDK's model object and recording every turn through
:func:`~flyteplugins.agents.core.durable_step`. Hermes is a template SDK that
exposes no concrete per-turn model hook, so this adapter records durability at
the *run* granularity instead: the whole ``agent.run`` is recorded as one
``flyte.trace`` leaf keyed by the request identity, so a crashed/retried Flyte
task replays the recorded answer (and the durable tool-call child actions) rather
than re-driving — and re-billing — the model.

If Hermes later grows a model interface, swap this for a per-turn wrapper that
mirrors ``openai._durable.FlyteModel``.
"""

from __future__ import annotations

import typing

from flyteplugins.agents.core import durable_step, fingerprint, jsonable


def _request_fingerprint(
    *,
    input: str,
    history: typing.Sequence[dict[str, typing.Any]],
    tools: typing.Sequence[typing.Any],
    model: str | None,
    instructions: str | None,
) -> str:
    """Deterministic memo key for an agent run — tool names, not callables."""
    return fingerprint(
        {
            "input": input,
            "history": jsonable(list(history)),
            "model": model,
            "instructions": instructions,
            "tools": sorted(getattr(t, "__name__", getattr(t, "name", str(t))) for t in (tools or [])),
        }
    )


async def durable_run(
    run: typing.Callable[[], typing.Awaitable[typing.Any]],
    *,
    input: str,
    history: typing.Sequence[dict[str, typing.Any]] = (),
    tools: typing.Sequence[typing.Any] = (),
    model: str | None = None,
    instructions: str | None = None,
) -> str:
    """Run ``run()`` once as a durable, replayable step keyed by the request identity.

    ``run`` performs the real (non-serializable) ``agent.run`` call; only the
    serializable request key is fed to the trace. The recorded value is the final
    answer string, replayed on retry.
    """
    return await durable_step(
        _request_fingerprint(input=input, history=history, tools=tools, model=model, instructions=instructions),
        lambda: _as_text(run()),
        name="agent_run",
        dumps=lambda text: text,
        loads=lambda text: text,
    )


async def _as_text(awaitable: typing.Awaitable[typing.Any]) -> str:
    result = await awaitable
    return str(result) if result is not None else ""
