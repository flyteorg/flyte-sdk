"""
Sandbox Configuration Options
==============================

Sandboxed tasks accept configuration for timeouts, memory limits, stack
depth, and type checking. These settings control the Monty sandbox
environment.

You can also use standard Flyte task options like ``cache`` and ``retries``.
Both ``def`` and ``async def`` functions are supported with all options.

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

from typing import List

import flyte
import flyte.sandbox

# --- Custom timeout -----------------------------------------------------------
# Default is 30 seconds. Use a shorter timeout for tasks that should be fast,
# or a longer one for complex computations.


@flyte.sandbox.orchestrator(timeout_ms=5_000)
def quick_sum(numbers: List[int]) -> int:
    total = 0
    for n in numbers:
        total = total + n
    return total


# --- Memory limit -------------------------------------------------------------
# Default is 50 MB. Increase for tasks that work with large data structures.


@flyte.sandbox.orchestrator(max_memory=100 * 1024 * 1024)  # 100 MB
def build_large_list(n: int) -> List[int]:
    result = []
    for i in range(n):
        result.append(i * i)
    return result


# --- Stack depth --------------------------------------------------------------
# Default is 256. Increase for deeply recursive computations.


@flyte.sandbox.orchestrator(max_stack_depth=512)
def deep_recursion(n: int) -> int:
    if n <= 1:
        return 1
    return n + deep_recursion(n - 1)


# --- Disable type checking ----------------------------------------------------
# By default Monty validates types at the boundary. Disable for flexibility.


@flyte.sandbox.orchestrator(type_check=False)
def flexible(x: int, y: int) -> int:
    return x + y


# --- Caching ------------------------------------------------------------------
# Enable caching so repeated calls with the same inputs skip re-execution.


@flyte.sandbox.orchestrator(cache="auto")
def expensive_calc(x: int) -> int:
    # Imagine this is expensive — caching avoids redundant work.
    result = x
    for _ in range(1000):
        result = (result * 31 + 17) % 1000000007
    return result


# --- Retries ------------------------------------------------------------------
# Automatically retry on failure.


@flyte.sandbox.orchestrator(retries=3)
def flaky_task(x: int) -> int:
    return x * 2


# --- Custom name --------------------------------------------------------------
# Override the auto-generated task name for clarity in the Flyte UI.


@flyte.sandbox.orchestrator(name="my-adder")
def adder(x: int, y: int) -> int:
    return x + y


# --- Combining options --------------------------------------------------------


@flyte.sandbox.orchestrator(
    timeout_ms=10_000,
    max_memory=25 * 1024 * 1024,
    cache="auto",
    retries=2,
    name="robust-transform",
)
def robust_transform(values: List[int], factor: int) -> List[int]:
    result = []
    for v in values:
        result.append(v * factor)
    return result


# --- Async with configuration -------------------------------------------------
# All configuration options work with ``async def`` functions too.


@flyte.sandbox.orchestrator(timeout_ms=5_000, cache="auto")
async def async_sum(numbers: List[int]) -> int:
    total = 0
    for n in numbers:
        total = total + n
    return total


# --- orchestrator_from_str() with configuration -------------------------------
# The same options work with ``orchestrator_from_str()`` for code-string tasks.

configured_code_task = flyte.sandbox.orchestrator_from_str(
    "x * factor",
    inputs={"x": int, "factor": int},
    output=int,
    name="configured-multiply",
    timeout_ms=5_000,
    cache="auto",
    retries=1,
)


# --- Attach to an environment for ``flyte run`` -----------------------------

sandbox_env = flyte.TaskEnvironment.from_task(
    "config-demo",
    quick_sum,
    build_large_list,
    deep_recursion,
    flexible,
    expensive_calc,
    flaky_task,
    adder,
    robust_transform,
    async_sum,
    configured_code_task,
)


if __name__ == "__main__":
    print("quick_sum([1,2,3]) =", quick_sum.forward([1, 2, 3]))
    print("build_large_list(5) =", build_large_list.forward(5))
    print("deep_recursion(10) =", deep_recursion.forward(10))
    print("flexible(2, 3) =", flexible.forward(2, 3))
    print("expensive_calc(42) =", expensive_calc.forward(42))
    print("flaky_task(7) =", flaky_task.forward(7))
    print("adder(1, 2) =", adder.forward(1, 2))
    print("robust_transform([1,2,3], 10) =", robust_transform.forward([1, 2, 3], 10))
    print("async_sum([1,2,3,4,5]) =", async_sum.forward([1, 2, 3, 4, 5]))
