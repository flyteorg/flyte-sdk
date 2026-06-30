"""
Sandboxed Orchestrator
======================

A sandboxed task can call regular ``env.task`` workers. When the sandbox
hits an external call, Monty **pauses**, the Flyte controller dispatches
the worker in its own container, and Monty **resumes** with the result.

This means the orchestrator body is pure Python (cheap, fast, side-effect
free) while the heavy lifting runs in full containers with filesystem
and network access.

Both ``def`` and ``async def`` orchestrators are supported — Monty natively
handles ``await`` expressions.

Use ``@env.sandbox.orchestrator`` to define sandboxed tasks directly on a
``TaskEnvironment``, so they share the environment's image and are
automatically registered for ``flyte run``.

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

import flyte

env = flyte.TaskEnvironment(
    name="orchestrator-demo", image=flyte.Image.from_debian_base().with_pip_packages("pydantic-monty")
)


# --- Worker tasks (run in their own containers) ------------------------------


@env.task
def fetch_score(player_id: int) -> int:
    """Simulate fetching a score from a database."""
    # In real code this would hit a DB or API.
    scores = {1: 42, 2: 87, 3: 15, 4: 63, 5: 99}
    return scores.get(player_id, 0)


@env.task
def multiply(x: int, y: int) -> int:
    return x * y


@env.task
def add(x: int, y: int) -> int:
    return x + y


# --- Sandboxed orchestrator --------------------------------------------------
# The orchestrator contains only control flow. Each call to a worker task
# pauses the sandbox, runs the worker, and resumes with the result.
# Using ``@env.sandbox.orchestrator`` registers the task directly in the environment.


@env.sandbox.orchestrator
def leaderboard(player_ids: list[int]) -> dict[str, int]:
    """Compute total and bonus scores for a list of players."""
    total = 0
    best = 0
    for pid in player_ids:
        score = fetch_score(pid)
        total = add(total, score)
        if score > best:
            best = score

    # Bonus = best score doubled
    bonus = multiply(best, 2)

    return {
        "total": total,
        "best": best,
        "bonus": bonus,
    }


# --- Chained orchestration ---------------------------------------------------
# Sandboxed tasks can compose multiple workers into a pipeline.


@env.sandbox.orchestrator
def scaled_sum(a: int, b: int, scale: int) -> int:
    """Add two numbers, then multiply by a scale factor."""
    raw = add(a, b)
    return multiply(raw, scale)


# --- Async orchestrator -------------------------------------------------------
# ``async def`` orchestrators are fully supported. Monty drives the coroutine
# natively, so ``await`` works inside the sandbox.


@env.sandbox.orchestrator
async def async_leaderboard(player_ids: list[int]) -> dict[str, int]:
    """Same as leaderboard, but defined as an async function."""
    total = 0
    best = 0
    for pid in player_ids:
        score = fetch_score(pid)
        total = add(total, score)
        if score > best:
            best = score
    bonus = multiply(best, 2)
    return {
        "total": total,
        "best": best,
        "bonus": bonus,
    }


if __name__ == "__main__":
    # forward() calls the Python function directly — external calls
    # resolve to the worker's forward() as well.
    print("leaderboard([1,2,3]) =", leaderboard.forward([1, 2, 3]))
    print("scaled_sum(3, 4, 10) =", scaled_sum.forward(3, 4, 10))
    print("async_leaderboard([1,2,3]) =", async_leaderboard.forward([1, 2, 3]))
