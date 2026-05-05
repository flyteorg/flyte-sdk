"""
Run a small fanout against the Rust-backed RemoteController.

The Rust controller (`flyte_controller_base`) is opt-in via the
`_F_USE_RUST_CONTROLLER=1` environment variable. With the variable unset,
the SDK uses the pure-Python `RemoteController` — this example will still
work, just on the Python path.

Selection rules:

    _F_USE_RUST_CONTROLLER  _U_USE_ACTIONS   wheel installed   selected
    ----------------------  --------------   ---------------   -----------------
    unset / 0               any              any               Python controller
    1                       1                any               Python controller (Actions service is Python-only)
    1                       unset / 0        yes               Rust controller
    1                       unset / 0        no                Python controller (warning logged)

Quickstart::

    # 1. Build the wheel and install it (one-time, see README "Developing
    #    the Rust Core Controller"):
    cd rs_controller && make build-wheel-local && cd ..
    uv pip install --find-links ./rs_controller/dist --no-index \\
        --force-reinstall --no-deps flyte_controller_base

    # 2. Opt in and run.
    _F_USE_RUST_CONTROLLER=1 python examples/advanced/rust_controller.py

The example does a 50-way fanout of a trivial task to give the Rust
informer/submit loop something to chew on. Bump `FAN_OUT` in your
environment to push it harder.
"""

from __future__ import annotations

import asyncio
import os
import time

import flyte

env = flyte.TaskEnvironment(
    name="rust_controller_demo",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)


@env.task
async def square(i: int) -> int:
    """Trivial leaf task — purpose is to be cheap so the controller's
    submit/watch loop is the bottleneck, not the work."""
    return i * i


@env.task
async def fanout(n: int) -> int:
    """Submit `n` parallel sub-actions and return the sum of squares."""
    print(f"[fanout] submitting {n} sub-actions")
    t0 = time.monotonic()
    results = await asyncio.gather(*[square(i=i) for i in range(n)])
    elapsed = time.monotonic() - t0
    total = sum(results)
    print(f"[fanout] {n} sub-actions completed in {elapsed:.2f}s, sum={total}")
    return total


if __name__ == "__main__":
    fan_out = int(os.getenv("FAN_OUT", "50"))
    using_rust = os.getenv("_F_USE_RUST_CONTROLLER") == "1"
    print(f"Controller path: {'Rust' if using_rust else 'Python'} (FAN_OUT={fan_out})")

    flyte.init_from_config()
    run = flyte.run(fanout, n=fan_out)
    print(f"run name: {run.name}")
    print(f"run url:  {run.url}")
    run.wait()
