"""
Reusable Tasks from Code Strings
=================================

``flyte.sandboxed.code()`` creates a reusable ``CodeTaskTemplate`` from a
Python code string. The returned template works with ``flyte.run()`` just
like a decorated task — but the source is a string, not a function.

This is useful when:
- Code is generated dynamically (e.g. from a UI or LLM)
- You want to define lightweight pipelines without writing decorated functions
- You need to ship a code snippet to a sandboxed worker

**Return value convention**: The value of the **last expression** in the
code becomes the return value. No ``return`` statement needed.

Install the optional dependency first::

    pip install 'flyte[sandboxed]'
"""

import flyte
import flyte.sandboxed

env = flyte.TaskEnvironment(name="code-string-demo")


# --- Worker tasks available inside the sandbox -------------------------------

@env.task
def add(x: int, y: int) -> int:
    return x + y


@env.task
def multiply(x: int, y: int) -> int:
    return x * y


# --- Example 1: Simple expression -------------------------------------------
# A single expression — its value is the return value.

double = flyte.sandboxed.code(
    "x * 2",
    inputs={"x": int},
    output=int,
    name="double",
)
# flyte.run(double, x=5)  → 10


# --- Example 2: Multi-line with assignment -----------------------------------
# When the last statement is an assignment, the assigned variable is returned.

scale_and_offset = flyte.sandboxed.code(
    """
    scaled = x * factor
    result = scaled + offset
    """,
    inputs={"x": int, "factor": int, "offset": int},
    output=int,
    name="scale-and-offset",
)
# flyte.run(scale_and_offset, x=10, factor=3, offset=5)  → 35


# --- Example 3: Calling external tasks --------------------------------------
# Pass worker tasks via ``functions={}`` so the sandbox can call them.

compute_pipeline = flyte.sandboxed.code(
    """
    partial = add(x, y)
    result = multiply(partial, scale)
    """,
    inputs={"x": int, "y": int, "scale": int},
    output=int,
    functions={"add": add, "multiply": multiply},
    name="compute-pipeline",
)
# flyte.run(compute_pipeline, x=2, y=3, scale=4)  → 20


# --- Example 4: String processing -------------------------------------------
# Sandboxed code can work with strings and collections too.

format_greeting = flyte.sandboxed.code(
    """
    parts = []
    for name in names:
        parts.append(greeting + ", " + name + "!")
    result = "; ".join(parts)
    """,
    inputs={"greeting": str, "names": list},
    output=str,
    name="format-greeting",
)
# flyte.run(format_greeting, greeting="Hello", names=["Alice", "Bob"])
# → "Hello, Alice!; Hello, Bob!"


# --- Example 5: Conditional logic -------------------------------------------

classify = flyte.sandboxed.code(
    """
    if score >= 90:
        label = "A"
    elif score >= 80:
        label = "B"
    elif score >= 70:
        label = "C"
    else:
        label = "F"
    """,
    inputs={"score": int},
    output=str,
    name="classify-score",
)
# flyte.run(classify, score=85)  → "B"


# --- Example 6: Building a dict result --------------------------------------

summarize = flyte.sandboxed.code(
    """
    total = 0
    count = 0
    for v in values:
        total = total + v
        count = count + 1
    result = {"total": total, "count": count, "avg": total / count}
    """,
    inputs={"values": list},
    output=dict,
    name="summarize",
)
# flyte.run(summarize, values=[10, 20, 30])  → {"total": 60, "count": 3, "avg": 20.0}


# --- Example 7: No output (side-effect-free validation) ----------------------
# Omit ``output=`` to get NoneType — useful for pure validation logic.

validate = flyte.sandboxed.code(
    """
    for item in items:
        if item < 0:
            raise ValueError("negative value: " + str(item))
    """,
    inputs={"items": list},
    name="validate-positive",
)
# flyte.run(validate, items=[1, 2, 3])  → None (no error)


# --- Attach to an environment for ``flyte run`` -----------------------------
# ``code()`` creates standalone templates. ``flyte run`` requires every task
# to belong to a TaskEnvironment.

sandbox_env = flyte.TaskEnvironment.from_task(
    "code-string-demo",
    double,
    scale_and_offset,
    compute_pipeline,
    format_greeting,
    classify,
    summarize,
    validate,
)
