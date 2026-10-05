"""
What an orchestrator sees when it hits a limit
==============================================

A sandbox leaseworker runs many orchestrators in one process, so it limits
what one of them can do. Each orchestrator here breaks one limit; running
this file prints the error the run ends with, as ``flyte`` reports it and
as the UI shows it on the run's attempt::

    python limits.py

The limits, and the error a run gets for each:

- ``timeout_ms`` on the decorator (at most the worker's limit, 30 s by
  default): time the source spends executing, not counting time waiting for
  tasks. ``TimeoutError``.
- ``max_stack_depth`` on the decorator (at most 256): ``RecursionError``.
- Memory, set on the worker (256 MiB by default) and the same for every
  orchestrator: ``MemoryError``, with what to do instead. An orchestrator
  only passes small values between tasks; large data belongs inside a task.
  A run that happens to be executing at the moment another goes over the
  limit can be stopped too; it fails with ``WorkerMemoryPressure``, a system
  error that is retried, and shows a second attempt.
- Task calls per run, set on the worker (10,000 by default):
  ``TooManyTaskCalls``. The source cannot catch it.
- Values passed to and returned from tasks must be plain data (None, bool,
  int, float, str, list, tuple, string-keyed dict) nested no deeper than the
  interpreter allows: ``TypeError``.
- Error messages are kept to their first 10 KiB.

All of these are user errors and are not retried, except
``WorkerMemoryPressure``. A task that fails is different: its error is raised
in the source, where it can be caught (see ``tolerant`` in
``orchestrator.py``).
"""

import flyte
import flyte.remote
import flyte.sandbox


@flyte.sandbox.orchestrator
def square(x: int) -> int:
    return x * x


@flyte.sandbox.orchestrator(timeout_ms=2_000)
def spins(x: int) -> int:
    while True:
        x = x + 1


@flyte.sandbox.orchestrator(max_stack_depth=50)
def recurses(x: int) -> int:
    def depth(n):
        return depth(n + 1)

    return depth(x)


@flyte.sandbox.orchestrator
def allocates(x: int) -> int:
    return len("a" * (x << 30))


@flyte.sandbox.orchestrator
def calls_forever(x: int) -> int:
    total = 0
    while True:
        total = total + square(x)


@flyte.sandbox.orchestrator
def passes_deep_value(x: int) -> int:
    value = []
    for _ in range(x):
        value = [value]
    return square(value)


@flyte.sandbox.orchestrator
def raises_long_error(x: int) -> int:
    raise ValueError("details " * x)


# calls_forever is left out: at the worker's default limit it launches 10,000
# child orchestrators before it fails. Add it to see TooManyTaskCalls.
CASES = [
    (spins, {"x": 0}),
    (recurses, {"x": 0}),
    (allocates, {"x": 16}),
    (passes_deep_value, {"x": 100_000}),
    (raises_long_error, {"x": 1_000_000}),
]


if __name__ == "__main__":
    flyte.init_from_config()
    runs = [(task, flyte.run(task, **inputs)) for task, inputs in CASES]
    for task, run in runs:
        run.wait(quiet=True)
        details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
        print(f"{task.name}: {details.phase.name} after {len(details.pb2.attempts)} attempt(s)  {run.url}")
        for attempt in details.pb2.attempts:
            error = attempt.error_info
            message = error.message
            if len(message) > 200:
                message = f"{message[:200]}... ({len(message)} chars)"
            print(f"  attempt {attempt.attempt}: {error.code} ({error.Kind.Name(error.kind)}): {message}")
