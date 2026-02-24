"""Sandbox utilities for running isolated code inside Flyte tasks.

.. warning:: Experimental feature: alpha — APIs may change without notice.

``flyte.sandbox`` provides two distinct sandboxing approaches:

---

**1. Orchestration sandbox** — powered by Monty
    Runs pure Python *orchestration logic* (control flow, routing, aggregation)
    with zero overhead. The Monty runtime enforces strong restrictions:
    no imports, no IO, no network access, microsecond startup.  Used via
    ``@env.sandbox.orchestrator`` or ``flyte.sandbox.orchestrator_from_str()``.

    Sandboxed orchestrators are:

    - **Side-effect free**: No filesystem, network, or OS access
    - **Microsecond startup**: No container spin-up — runs in the same process
    - **Multiplexable**: Many orchestrators run safely on the same Python process

    Example::

        env = flyte.TaskEnvironment(name="my-env")

        @env.sandbox.orchestrator
        def route(x: int, y: int) -> int:
            return add(x, y)   # calls a worker task

        pipeline = flyte.sandbox.orchestrator_from_str(
            "add(x, y) * 2",
            inputs={"x": int, "y": int},
            output=int,
            tasks=[add],
        )

---

**2. Code sandbox** — arbitrary code in an isolated container
    Runs arbitrary Python scripts or shell commands inside an ephemeral Docker
    container. The image is built on demand from declared ``packages`` and
    ``system_packages``, executed once, then discarded. Network is blocked by
    default (``block_network=True``), preventing outbound calls from untrusted
    code.  Used via ``flyte.sandbox.create()``.

    Code sandboxes support:

    - **Arbitrary pip packages** — install any Python dependency at runtime
    - **System packages** — apt packages, compilers, native libs
    - **Code mode** — run a Python snippet with typed inputs/outputs
    - **Command mode** — run any shell command (e.g. ``pytest``, a binary)

    Example — code mode::

        sandbox = flyte.sandbox.create(
            name="compute",
            code=\"\"\"
                import argparse, pathlib
                parser = argparse.ArgumentParser()
                parser.add_argument("--x", type=int)
                args = parser.parse_args()
                pathlib.Path("/var/outputs/result").write_text(str(args.x * 2))
            \"\"\",
            inputs={"x": int},
            outputs={"result": int},
        )
        (result,) = await sandbox.run.aio(x=21)   # returns (42,)

    Example — command mode::

        sandbox = flyte.sandbox.create(
            name="test-runner",
            command=["/bin/bash", "-c", "pytest /var/inputs/tests.py -q"],
            inputs={"tests.py": File},
            outputs={"exit_code": str},
        )
"""

from ._api import orchestrate_local, orchestrator_from_str
from ._code_sandbox import ImageConfig, create, sandbox_environment
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
    "ImageConfig",
    "SandboxedConfig",
    "SandboxedTaskTemplate",
    "create",
    "orchestrate_local",
    "orchestrator_from_str",
    "sandbox_environment",
]
