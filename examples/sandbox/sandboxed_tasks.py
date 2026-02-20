"""
Sandboxed tasks run pure Python in a Monty sandbox — Pydantic's Rust-based
Python interpreter. They are side-effect free (no filesystem, network, or OS
access), start in microseconds, and many can multiplex safely on a single
container process.

In this example the orchestrator is a sandboxed task that calls regular
``env.task`` workers. Use ``@env.sandbox.orchestrate`` to register sandboxed
tasks directly in a ``TaskEnvironment``, so they share the environment's
image and are ready for ``flyte run`` without extra boilerplate.

Install the optional dependency first:

    pip install 'flyte[sandbox]'
"""

import flyte
import flyte.sandbox

env = flyte.TaskEnvironment(name="sandboxed-demo")


# --- Regular tasks — run in their own containers ---------------------------


@env.task
def add(x: int, y: int) -> int:
    return x + y


@env.task
def multiply(x: int, y: int) -> int:
    return x * y


@env.task
def fib(n: int) -> int:
    """Compute the nth Fibonacci number iteratively."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# --- Sandboxed orchestrator ------------------------------------------------
# The orchestrator is a sandboxed task registered directly in the environment.
# It contains only pure Python control flow — all heavy lifting is dispatched
# to the regular tasks above.


@env.sandbox.orchestrate
def pipeline(n: int) -> dict[str, int]:
    fib_result = fib(n)
    linear_result = add(multiply(n, 2), 5)
    total = add(fib_result, linear_result)

    return {
        "fib": fib_result,
        "linear": linear_result,
        "total": total,
    }


# --- orchestrate(): reusable task from a code string ----------------------
# Instead of defining a @sandbox.task function, send a code string directly.
# External tasks are provided via the ``tasks`` list.

code_pipeline = flyte.sandbox.orchestrate(
    """
    partial = add(x, y)
    result = partial * 2
    """,
    inputs={"x": int, "y": int},
    output=int,
    tasks=[add],
    name="code-pipeline",
)
# Use with: flyte.run(code_pipeline, x=1, y=2)  → 6


# --- orchestrate_local(): one-shot code execution --------------------------
# For quick one-off computations, ``orchestrate_local()`` sends code + inputs
# and returns the result directly (no TaskTemplate, no controller).
#
#   result = await flyte.sandbox.orchestrate_local(
#       "add(x, y) * 2",
#       inputs={"x": 1, "y": 2},
#       tasks=[add],
#   )
#   # → 6


# --- Attach code-string tasks to an environment for ``flyte run`` ---------
# ``orchestrate()`` creates standalone templates. Group them with
# ``from_task`` so they belong to a TaskEnvironment.

sandbox_env = flyte.TaskEnvironment.from_task(
    "sandboxed-orchestrator",
    code_pipeline,
)


if __name__ == "__main__":
    # Quick local verification using forward() — bypasses Monty, calls Python directly.
    print("add(2, 3) =", add.forward(2, 3))
    print("multiply(4, 5) =", multiply.forward(4, 5))
    print("fib(10) =", fib.forward(10))
    print("pipeline(10) =", pipeline.forward(10))
