"""
One-Shot Code Execution with ``orchestrate_local()``
=====================================================

``flyte.sandbox.orchestrate_local()`` sends a code string + inputs to
the Monty sandbox and returns the result directly. No ``TaskTemplate``,
no controller, no registration — just *send code, get answer*.

This is the quickest way to evaluate an expression or snippet in the
sandbox.  It's ``async`` because external function calls (tasks, durable
ops) require ``await``.

For reusable tasks from code strings, see ``orchestrator()`` which
creates a ``CodeTaskTemplate`` that can be passed to ``flyte.run()``.

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

import asyncio

import flyte
import flyte.sandbox

env = flyte.TaskEnvironment(name="oneshot-demo")


@env.task
def add(x: int, y: int) -> int:
    return x + y


@env.task
def power(base: int, exp: int) -> int:
    result = 1
    for _ in range(exp):
        result = result * base
    return result


async def main():
    # --- Pure Python: simple expression --------------------------------------
    result = await flyte.sandbox.orchestrate_local(
        "x + y",
        inputs={"x": 1, "y": 2},
    )
    print(f"x + y = {result}")  # 3

    # --- Pure Python: multi-line snippet -------------------------------------
    result = await flyte.sandbox.orchestrate_local(
        """
        total = 0
        for i in range(n):
            total = total + i
        total
        """,
        inputs={"n": 10},
    )
    print(f"sum(0..9) = {result}")  # 45

    # --- Pure Python: string manipulation ------------------------------------
    result = await flyte.sandbox.orchestrate_local(
        """
        words = text.split()
        " ".join(reversed(words))
        """,
        inputs={"text": "hello beautiful world"},
    )
    print(f"reversed words = {result}")  # "world beautiful hello"

    # --- Pure Python: list comprehension style (via loop) --------------------
    result = await flyte.sandbox.orchestrate_local(
        """
        squares = []
        for x in items:
            squares.append(x * x)
        squares
        """,
        inputs={"items": [1, 2, 3, 4, 5]},
    )
    print(f"squares = {result}")  # [1, 4, 9, 16, 25]

    # --- With external tasks -------------------------------------------------
    # When ``tasks=`` is provided, external calls pause Monty, dispatch
    # the real task, and resume with the result.
    result = await flyte.sandbox.orchestrate_local(
        "add(x, y) * 2",
        inputs={"x": 3, "y": 4},
        tasks=[add],
    )
    print(f"add(3, 4) * 2 = {result}")  # 14

    # --- Chaining external tasks ---------------------------------------------
    result = await flyte.sandbox.orchestrate_local(
        """
        base_sum = add(a, b)
        power(base_sum, exp)
        """,
        inputs={"a": 2, "b": 3, "exp": 3},
        tasks=[add, power],
    )
    print(f"(2+3)^3 = {result}")  # 125

    # --- Mixing pure Python and external calls -------------------------------
    result = await flyte.sandbox.orchestrate_local(
        """
        results = []
        for i in range(n):
            results.append(add(i, 1))
        total = 0
        for r in results:
            total = total + r
        total
        """,
        inputs={"n": 5},
        tasks=[add],
    )
    print(f"sum(add(i,1) for i in 0..4) = {result}")  # 1+2+3+4+5 = 15


if __name__ == "__main__":
    asyncio.run(main())
