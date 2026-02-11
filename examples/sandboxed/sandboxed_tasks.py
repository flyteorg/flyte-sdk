"""
Sandboxed tasks run pure Python in a Monty sandbox — Pydantic's Rust-based
Python interpreter. They are side-effect free (no filesystem, network, or OS
access), start in microseconds, and many can multiplex safely on a single
container process.

In this example the orchestrator is a sandboxed task that calls regular
env.task workers. When the sandboxed orchestrator calls another task,
Monty *pauses*, the Flyte controller dispatches the worker task in its
own container, and Monty *resumes* with the result.

Install the optional dependency first:

    pip install 'flyte[sandboxed]'
"""

import flyte
import flyte.sandboxed

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
# The orchestrator is a sandboxed task. It contains only pure Python
# control flow — all heavy lifting is dispatched to the regular tasks above.
# The sandboxed image includes pydantic-monty automatically.

@flyte.sandboxed.task
def pipeline(n: int) -> dict[str, int]:
    fib_result = fib(n)
    linear_result = add(multiply(n, 2), 5)
    total = add(fib_result, linear_result)

    return {
        "fib": fib_result,
        "linear": linear_result,
        "total": total,
    }


# --- code(): reusable task from a code string ------------------------------
# Instead of defining a @sandboxed.task function, send a code string directly.
# External tasks are provided via the ``functions`` dict.

code_pipeline = flyte.sandboxed.code(
    """
    partial = add(x, y)
    result = partial * 2
    """,
    inputs={"x": int, "y": int},
    output=int,
    functions={"add": add},
    name="code-pipeline",
)
# Use with: flyte.run(code_pipeline, x=1, y=2)  → 6


# --- run(): one-shot code execution ----------------------------------------
# For quick one-off computations, ``run()`` sends code + inputs and returns
# the result directly (no TaskTemplate, no controller).
#
#   result = await flyte.sandboxed.run(
#       "add(x, y) * 2",
#       inputs={"x": 1, "y": 2},
#       functions={"add": add},
#   )
#   # → 6


# --- Attach sandboxed tasks to an environment for ``flyte run`` -----------
# ``@flyte.sandboxed.task`` creates standalone templates.  ``flyte run``
# requires every task to belong to a TaskEnvironment.  Group them with
# ``from_task`` so they share the same sandboxed image.

sandbox_env = flyte.TaskEnvironment.from_task(
    "sandboxed-orchestrator",
    pipeline,
    code_pipeline,
)


if __name__ == "__main__":
    # Quick local verification using forward() — bypasses Monty, calls Python directly.
    print("add(2, 3) =", add.forward(2, 3))
    print("multiply(4, 5) =", multiply.forward(4, 5))
    print("fib(10) =", fib.forward(10))
    print("pipeline(10) =", pipeline.forward(10))
