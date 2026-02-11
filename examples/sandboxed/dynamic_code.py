"""
Dynamically Generated Code
===========================

Because ``flyte.sandboxed.code()`` and ``flyte.sandboxed.run()`` accept
plain strings, the code can be generated at runtime — from templates,
user input, or even LLM output.

This example shows patterns for building code strings programmatically
and executing them in the sandbox.

Install the optional dependency first::

    pip install 'flyte[sandboxed]'
"""

import asyncio

import flyte
import flyte.sandboxed

env = flyte.TaskEnvironment(name="dynamic-demo")


@env.task
def add(x: int, y: int) -> int:
    return x + y


# --- Pattern 1: Parameterized code templates ---------------------------------
# Build code strings from templates based on runtime configuration.

def make_reducer(operation: str) -> flyte.sandboxed.CodeTaskTemplate:
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

    return flyte.sandboxed.code(
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
) -> flyte.sandboxed.CodeTaskTemplate:
    """Create a task that evaluates a formula string.

    Example::

        t = make_formula_task("a * b + c", ["a", "b", "c"])
        flyte.run(t, a=2, b=3, c=4)  # → 10
    """
    return flyte.sandboxed.code(
        formula,
        inputs={v: float for v in variables},
        output=float,
        name=f"formula-{formula.replace(' ', '')}",
    )


area_of_triangle = make_formula_task("0.5 * base * height", ["base", "height"])
# flyte.run(area_of_triangle, base=10.0, height=5.0)  → 25.0

bmi = make_formula_task("weight / (height * height)", ["weight", "height"])
# flyte.run(bmi, weight=70.0, height=1.75)  → ~22.86


# --- Pattern 3: run() for ad-hoc evaluation ---------------------------------

async def evaluate_expressions():
    """Evaluate a batch of expressions in the sandbox."""
    expressions = [
        ("2 ** 10", {}),
        ("x * x + y * y", {"x": 3, "y": 4}),
        ("len(items)", {"items": [1, 2, 3, 4, 5]}),
    ]

    for expr, extra_inputs in expressions:
        inputs = {**extra_inputs} if extra_inputs else {"_unused": 0}
        # For expressions that don't reference any inputs, we still need
        # at least one input for Monty.
        if not extra_inputs:
            result = await flyte.sandboxed.run(expr, inputs={"_unused": 0})
        else:
            result = await flyte.sandboxed.run(expr, inputs=extra_inputs)
        print(f"  {expr} = {result}")


# --- Pattern 4: Composing code with external tasks ---------------------------
# Generate orchestration code that calls worker tasks.

def make_map_reduce(
    map_task_name: str,
    reduce_op: str,
    map_task,
) -> flyte.sandboxed.CodeTaskTemplate:
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
    return flyte.sandboxed.code(
        code,
        inputs={"items": list, "factor": int},
        output=int,
        functions={map_task_name: map_task},
        name=f"map-{map_task_name}-reduce-{reduce_op}",
    )


@env.task
def scale(x: int, factor: int) -> int:
    return x * factor


map_scale_sum = make_map_reduce("scale", "+", scale)
# flyte.run(map_scale_sum, items=[1, 2, 3], factor=10)
# → scale(1,10) + scale(2,10) + scale(3,10) = 10 + 20 + 30 = 60


# --- Attach to an environment for ``flyte run`` -----------------------------

sandbox_env = flyte.TaskEnvironment.from_task(
    "dynamic-code-demo",
    sum_task,
    product_task,
    max_task,
    area_of_triangle,
    bmi,
    map_scale_sum,
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
