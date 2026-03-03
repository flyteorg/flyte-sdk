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

    Three execution modes are supported:

    - Code mode — provide Python source that runs with automatic input/output wiring.
    - Verbatim mode — run a script that manages its own I/O via /var/inputs and /var/outputs.
    - Command mode — execute an arbitrary command or entrypoint.

    Examples
    --------

    Code mode
    ~~~~~~~~~

    Provide Python code that uses inputs as variables and assigns
    outputs as Python values.

        _stats_code = \"""
        import numpy as np
        nums = np.array([float(v) for v in values.split(",")])
        mean = float(np.mean(nums))
        std  = float(np.std(nums))

        window_end = dt + delta
        \"""

        stats_sandbox = flyte.sandbox.create(
            name="numpy-stats",
            code=_stats_code,
            inputs={
                "values": str,
                "dt": datetime.datetime,
                "delta": datetime.timedelta,
            },
            outputs={
                "mean": float,
                "std": float,
                "window_end": datetime.datetime,
            },
            packages=["numpy"],
        )

        mean, std, window_end = await stats_sandbox.run.aio(
            values="1,2,3,4,5",
            dt=datetime.datetime(2024, 1, 1),
            delta=datetime.timedelta(days=1),
        )


    Verbatim mode
    ~~~~~~~~~~~~~

    Run a script that explicitly reads inputs from /var/inputs and
    writes outputs to /var/outputs.

        _etl_script = \"\"\"\
        import json, pathlib

        payload = json.loads(
            pathlib.Path("/var/inputs/payload").read_text()
        )
        total = sum(payload["values"])

        pathlib.Path("/var/outputs/total").write_text(str(total))
        \"\"\"

        etl_sandbox = flyte.sandbox.create(
            name="etl-script",
            code=_etl_script,
            inputs={"payload": File},
            outputs={"total": int},
            auto_io=False,
        )


    Command mode
    ~~~~~~~~~~~~

    Execute an arbitrary command inside the sandbox environment.

        sandbox = flyte.sandbox.create(
            name="test-runner",
            command=["/bin/bash", "-c", "pytest /var/inputs/tests.py -q"],
            inputs={"tests.py": File},
            outputs={"exit_code": str},
        )

    Notes
    -----

    • Inputs are materialized under /var/inputs.
    • Outputs must be written to /var/outputs.
    • In code mode, inputs are available as Python variables and
    scalar outputs are captured automatically.
    • Additional Python dependencies can be specified via the
    `packages` argument.
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
