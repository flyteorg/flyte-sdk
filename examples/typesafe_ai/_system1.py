"""System 1 — the TypeSafe model ("Jev").

A tiny async wrapper around the TypeSafe Python SDK.  Jev is a System One model:
give it a piece of state (raw text, a JSON object, ...) and a set of typed
questions (``Choice``/``Score``/``Noul``) and it answers with calibrated
probabilities, not generated prose.  The wrapper captures the typed answers plus
token usage and wall-clock latency so the benchmark can tell the "Jev is fast and
cheap" story quantitatively.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class JevDecision:
    """Typed answers + metadata for one TypeSafe ``system_one`` call."""

    choices: dict = field(default_factory=dict)  # question -> selected label
    scores: dict = field(default_factory=dict)  # question -> score (int)
    nouls: dict = field(default_factory=dict)  # question -> p(yes) (float 0..1)
    confidence: dict = field(default_factory=dict)  # question -> confidence 0..1
    probabilities: dict = field(default_factory=dict)  # question -> {label: p}
    n_questions: int = 0  # how many questions this single call answered
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


def _make_client():
    # Imported lazily so task images that only need one half of the stack stay light.
    from typesafe_sdk import AsyncTypeSafeClient

    return AsyncTypeSafeClient()


async def system_one(state, questions, *, client=None, model=None) -> JevDecision:
    """Run one Jev decision.

    ``questions`` is a ``Mapping[question_name, Choice|Score|Noul]``.  Every
    question is evaluated against the same ``state`` in a *single* request, in
    parallel and in isolation — TypeSafe's **speculative fan-out**: "adding
    questions barely changes the response time", so the right move is to ask
    many small questions (including speculative ones you may not use) rather
    than one big one, and to compose the answers in code.
    """
    close = client is None
    client = client or _make_client()
    t0 = time.perf_counter()
    try:
        resp = await client.system_one(state=state, questions=questions, model=model)
    finally:
        lat = time.perf_counter() - t0
        if close:
            await client.aclose()

    d = JevDecision(latency_s=lat, model=resp.model)
    for name, ans in (resp.choices or {}).items():
        d.choices[name] = ans.choice
        d.confidence[name] = getattr(ans, "confidence", None)
        d.probabilities[name] = getattr(ans, "probabilities", None)
    for name, ans in (resp.scores or {}).items():
        d.scores[name] = ans.score
        d.confidence[name] = getattr(ans, "confidence", None)
        d.probabilities[name] = getattr(ans, "probabilities", None)
    for name, ans in (resp.nouls or {}).items():
        d.nouls[name] = ans.noul
    d.n_questions = len(questions)
    if resp.usage is not None:
        d.input_tokens = resp.usage.input_tokens or 0
        d.output_tokens = resp.usage.output_tokens or 0
    return d


async def system_one_many(items, questions_for, *, concurrency=10):
    """Run many independent Jev decisions concurrently against one shared client.

    ``items`` -> iterable of ``(key, state)``; ``questions_for(key, state) ->
    questions mapping.  Returns ``{key: JevDecision}``.  Reusing a single client
    across the fan-out is far cheaper than opening a connection per decision.
    """
    client = _make_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(key, state):
        async with sem:
            return key, await system_one(state, questions_for(key, state), client=client)

    try:
        results = await asyncio.gather(*[one(k, s) for k, s in items])
    finally:
        await client.aclose()
    return dict(results)
