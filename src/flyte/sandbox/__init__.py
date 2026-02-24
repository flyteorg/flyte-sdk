"""Sandboxed tasks powered by Monty (Pydantic's Rust-based sandboxed Python interpreter).

.. warning:: Experimental feature: alpha — APIs may change without notice.

Sandboxed tasks are:
- **Side-effect free**: No filesystem, network, or OS access
- **Super fast**: Microsecond startup for pure Python
- **Multiplexable**: Many tasks run safely on the same Python process

Usage::

    import flyte
    import flyte.sandbox

    # Environment-based approach (preferred for ``flyte run``)
    env = flyte.TaskEnvironment(name="my-env")

    @env.sandbox.orchestrator
    def my_orchestrator(x: int, y: int) -> int:
        return add(x, y)

    # Create a reusable task from a code string
    pipeline = flyte.sandbox.orchestrator_from_str(
        "add(x, y) * 2",
        inputs={"x": int, "y": int},
        output=int,
        tasks=[add],
    )

    # One-shot execution of a code string (local only)
    result = await flyte.sandbox.orchestrate_local(
        "x + y",
        inputs={"x": 1, "y": 2},
    )
"""

from ._api import orchestrate_local, orchestrator_from_str
from ._code_task import CodeTaskTemplate
from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate

ORCHESTRATOR_SYNTAX_PROMPT = """\
CRITICAL — Sandbox syntax restrictions (Monty runtime):
- No imports allowed. All available functions are provided directly.
- No subscript assignment: `d[key] = value` and `l[i] = value` are FORBIDDEN.
- Reading subscripts is OK: `x = d[key]` and `x = l[i]` work fine.
- Build lists with .append() and list literals, NOT by index assignment.
- Build dicts ONLY as literals: {"k": v, ...}. Never mutate them after creation.
- To aggregate data, use lists of tuples/dicts, not mutating a dict.
- No `class` definitions.
- No `with` statements.
- No `try`/`except` blocks.
- No walrus operator (`:=`).
- No `yield` or `yield from` (generators).
- No `global` or `nonlocal` declarations.
- No set literals or set comprehensions.
- No `del` statements.
- No augmented assignment: `x += 1` is FORBIDDEN. Use `x = x + 1` instead.
- No `assert` statements.
- The last expression in your code is the return value.

Type restrictions:
- Allowed primitive types: int, float, str, bool, bytes, None.
- Allowed collection types: list, dict, tuple (including generic forms like list[int], dict[str, float]).
- Opaque IO handle types (pass-through only, cannot be inspected): File, Dir, DataFrame.
- Optional[T] and Union of allowed types are permitted.
- Custom classes, dataclasses, Pydantic models, and any other user-defined types are NOT allowed.
- set and frozenset are allowed as function parameter/return types but set literals and \
set comprehensions are not supported in code."""

__all__ = [
    "ORCHESTRATOR_SYNTAX_PROMPT",
    "CodeTaskTemplate",
    "SandboxedConfig",
    "SandboxedTaskTemplate",
    "orchestrate_local",
    "orchestrator_from_str",
]
