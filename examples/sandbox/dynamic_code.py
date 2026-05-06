"""
Dynamically Generated Code
===========================

Because ``flyte.sandbox.orchestrator_from_str()`` and
``flyte.sandbox.orchestrate_local()`` accept plain strings, the code
can be generated at runtime — from templates, user input, or even LLM
output.

This example shows patterns for building code strings programmatically
and executing them in the sandbox.

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

import asyncio

import flyte
import flyte.sandbox

env = flyte.TaskEnvironment(name="dynamic-demo")


@env.task
def add(x: int, y: int) -> int:
    return x + y


# --- Pattern 1: Parameterized code templates ---------------------------------
# Build code strings from templates based on runtime configuration.


def make_reducer(operation: str) -> flyte.sandbox.CodeTaskTemplate:
    """Create a sandboxed task that reduces a list using the given operation."""
    if operation == "sum":
        body = """
            acc = 0
            for v in values:
                acc = acc + v
            acc
        """
    elif operation == "product":
        body = """
            acc = 1
            for v in values:
                acc = acc * v
            acc
        """
    elif operation == "max":
        body = """
            acc = values[0]
            for v in values:
                if v > acc:
                    acc = v
            acc
        """
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return flyte.sandbox.orchestrator_from_str(
        body,
        inputs={"values": list},
        output=int,
        name=f"reduce-{operation}",
    )


sum_task = make_reducer("sum")
product_task = make_reducer("product")
max_task = make_reducer("max")
# flyte.run(sum_task, values=[1, 2, 3, 4])  → 10
# flyte.run(product_task, values=[1, 2, 3, 4])  → 24
# flyte.run(max_task, values=[1, 2, 3, 4])  → 4


# --- Pattern 2: Building formulas from user specs ----------------------------
# Imagine a config file or API that specifies a formula.


def make_formula_task(
    formula: str,
    variables: list[str],
) -> flyte.sandbox.CodeTaskTemplate:
    """Create a task that evaluates a formula string.

    Example::

        t = make_formula_task("a * b + c", ["a", "b", "c"])
        flyte.run(t, a=2, b=3, c=4)  # → 10
    """
    return flyte.sandbox.orchestrator_from_str(
        formula,
        inputs=dict.fromkeys(variables, float),
        output=float,
        name=f"formula-{formula.replace(' ', '')}",
    )


area_of_triangle = make_formula_task("0.5 * base * height", ["base", "height"])
# flyte.run(area_of_triangle, base=10.0, height=5.0)  → 25.0

bmi = make_formula_task("weight / (height * height)", ["weight", "height"])
# flyte.run(bmi, weight=70.0, height=1.75)  → ~22.86


# --- Pattern 3: orchestrate_local() for ad-hoc evaluation --------------------


async def evaluate_expressions():
    """Evaluate a batch of expressions in the sandbox."""
    expressions = [
        ("2 ** 10", {}),
        ("x * x + y * y", {"x": 3, "y": 4}),
        ("len(items)", {"items": [1, 2, 3, 4, 5]}),
    ]

    for expr, extra_inputs in expressions:
        # Monty requires at least one input, so pass a dummy for bare expressions.
        if not extra_inputs:
            result = await flyte.sandbox.orchestrate_local(expr, inputs={"_unused": 0})
        else:
            result = await flyte.sandbox.orchestrate_local(expr, inputs=extra_inputs)
        print(f"  {expr} = {result}")


# --- Pattern 4: Composing code with external tasks ---------------------------
# Generate orchestration code that calls worker tasks.


def make_map_reduce(
    map_task_name: str,
    reduce_op: str,
    map_task,
) -> flyte.sandbox.CodeTaskTemplate:
    """Build a map-reduce pipeline as sandboxed code."""
    code = f"""
mapped = []
for item in items:
    mapped.append({map_task_name}(item, factor))
acc = mapped[0]
for i in range(1, len(mapped)):
    acc = acc {reduce_op} mapped[i]
acc
"""
    return flyte.sandbox.orchestrator_from_str(
        code,
        inputs={"items": list, "factor": int},
        output=int,
        tasks=[map_task],
        name=f"map-{map_task_name}-reduce-{reduce_op}",
    )


@env.task
def scale(x: int, factor: int) -> int:
    return x * factor


map_scale_sum = make_map_reduce("scale", "+", scale)
# flyte.run(map_scale_sum, items=[1, 2, 3], factor=10)
# → scale(1,10) + scale(2,10) + scale(3,10) = 10 + 20 + 30 = 60


# --- Pattern 5: Parallel mapping with flyte_map ------------------------------
# ``flyte_map`` is a sandbox built-in that delegates to ``flyte.map``.
# It runs a task over an iterable in parallel and returns a list of results.
# Use it instead of sequential for-loops when you want concurrency.


@env.task
def square(x: int) -> int:
    return x * x


# flyte_map with a single iterable
parallel_squares = flyte.sandbox.orchestrator_from_str(
    """
    flyte_map("square", items)
    """,
    inputs={"items": list[int]},
    output=list[int],
    tasks=[square],
    name="parallel-squares",
)
# flyte.run(parallel_squares, items=[1, 2, 3, 4])  → [1, 4, 9, 16]


# flyte_map with multiple iterables (zipped) and concurrency limit
parallel_add = flyte.sandbox.orchestrator_from_str(
    """
    flyte_map("add", xs, ys, concurrency=4)
    """,
    inputs={"xs": list, "ys": list},
    output=list,
    tasks=[add],
    name="parallel-add",
)
# flyte.run(parallel_add, xs=[1, 2, 3], ys=[10, 20, 30])  → [11, 22, 33]


# flyte_map results fed into further sandbox logic
map_and_sum = flyte.sandbox.orchestrator_from_str(
    """
    squared = flyte_map("square", items)
    total = 0
    for v in squared:
        total = total + v
    total
    """,
    inputs={"items": list},
    output=int,
    tasks=[square],
    name="map-and-sum",
)
# flyte.run(map_and_sum, items=[1, 2, 3])  → 14  (1 + 4 + 9)


async def flyte_map_local():
    """Run flyte_map examples locally via orchestrate_local()."""
    # Basic parallel map
    result = await flyte.sandbox.orchestrate_local(
        'flyte_map("square", items)',
        inputs={"items": [1, 2, 3, 4, 5]},
        tasks=[square],
    )
    print(f"  squares: {result}")
    assert result == [1, 4, 9, 16, 25]

    # Multiple iterables
    result = await flyte.sandbox.orchestrate_local(
        'flyte_map("add", xs, ys)',
        inputs={"xs": [1, 2, 3], "ys": [10, 20, 30]},
        tasks=[add],
    )
    print(f"  pairwise add: {result}")
    assert result == [11, 22, 33]

    # Map then reduce
    result = await flyte.sandbox.orchestrate_local(
        """
        squared = flyte_map("square", items)
        total = 0
        for v in squared:
            total = total + v
        total
        """,
        inputs={"items": [1, 2, 3, 4]},
        tasks=[square],
    )
    print(f"  sum of squares: {result}")
    assert result == 30  # 1 + 4 + 9 + 16


# --- Attach to an environment for ``flyte run`` -----------------------------

sandbox_env = flyte.TaskEnvironment.from_task(
    "dynamic-code-demo",
    sum_task,
    product_task,
    max_task,
    area_of_triangle,
    bmi,
    map_scale_sum,
    parallel_squares,
    parallel_add,
    map_and_sum,
)


if __name__ == "__main__":
    print("--- Parameterized reducers ---")
    # forward() not available on CodeTaskTemplate, so we just verify creation
    print(f"sum_task: {sum_task.name}, inputs={sum_task._input_names}")
    print(f"product_task: {product_task.name}, inputs={product_task._input_names}")
    print(f"max_task: {max_task.name}, inputs={max_task._input_names}")

    print("\n--- Formula tasks ---")
    print(f"area_of_triangle: {area_of_triangle.name}, inputs={area_of_triangle._input_names}")
    print(f"bmi: {bmi.name}, inputs={bmi._input_names}")

    print("\n--- Ad-hoc expression evaluation ---")
    asyncio.run(evaluate_expressions())

    print("\n--- Map-reduce pipeline ---")
    print(f"map_scale_sum: {map_scale_sum.name}, has_external_refs={map_scale_sum._has_external_refs}")

    print("\n--- flyte_map local execution ---")
    asyncio.run(flyte_map_local())
