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

Quickstart (remote cluster)::

    # 1. Build the wheel for the cluster's architecture (multi-arch by default):
    cd rs_controller && make build-wheels && cd ..

    # 2. Install the wheel locally so the parent process can import it:
    uv pip install --find-links ./rs_controller/dist --no-index \\
        --force-reinstall --no-deps flyte_controller_base

    # 3. Opt in and run. The example automatically discovers
    #    ./rs_controller/dist and bakes the wheel into the task image
    #    via a PythonWheels layer, so child task containers also get
    #    the Rust controller wheel.
    _F_USE_RUST_CONTROLLER=1 python examples/advanced/rust_controller.py

To override the wheel location (e.g. shared CI artifacts dir)::

    FLYTE_RS_CONTROLLER_WHEELS=/path/to/wheels \\
        _F_USE_RUST_CONTROLLER=1 python examples/advanced/rust_controller.py

If no wheel directory is found the example falls back to the default
debian-base image — the Python controller is used end-to-end and the
opt-in becomes a no-op.

The example does a 50-way fanout of a trivial task to give the Rust
informer/submit loop something to chew on. Bump `FAN_OUT` in your
environment to push it harder.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import flyte
from flyte._image import PythonWheels


def _maybe_rust_controller_image() -> flyte.Image | None:
    """Build a debian-base image with the locally-built flyte_controller_base
    wheel installed, so task containers can pick up the Rust controller path.

    Resolution order for the wheel directory:

    1. ``FLYTE_RS_CONTROLLER_WHEELS`` env var — explicit override.
    2. ``rs_controller/dist`` relative to this file (works for in-tree
       development with ``make build-wheels`` or ``make build-wheel-local``).

    Returns ``None`` if no wheel directory is found, in which case we fall
    back to the default image. The example will still run on the Python
    controller path; the Rust opt-in will simply log a warning and use
    Python in the parent task.
    """
    override = os.getenv("FLYTE_RS_CONTROLLER_WHEELS")
    if override:
        wheel_dir = Path(override).expanduser().resolve()
    else:
        # examples/advanced/rust_controller.py -> repo_root/rs_controller/dist
        repo_root = Path(__file__).resolve().parent.parent.parent
        wheel_dir = repo_root / "rs_controller" / "dist"

    if not wheel_dir.is_dir() or not list(wheel_dir.glob("*.whl")):
        return None

    base = flyte.Image.from_debian_base()
    return base.clone(
        addl_layer=PythonWheels(wheel_dir=wheel_dir, package_name="flyte_controller_base")
    )


_image = _maybe_rust_controller_image()

env = flyte.TaskEnvironment(
    name="rust_controller_demo",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    # Only attach the custom image when we found a wheel; otherwise let
    # TaskEnvironment fall back to its default.
    **({"image": _image} if _image is not None else {}),
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
    wheel_attached = _image is not None

    print(f"Controller path:   {'Rust' if using_rust else 'Python'}")
    print(f"Wheel layer:       {'attached' if wheel_attached else 'not found (using default image)'}")
    print(f"FAN_OUT:           {fan_out}")
    if using_rust and not wheel_attached:
        print(
            "WARNING: _F_USE_RUST_CONTROLLER=1 was set but no wheel directory was "
            "found. Build wheels with `cd rs_controller && make build-wheels` or set "
            "FLYTE_RS_CONTROLLER_WHEELS to the directory containing flyte_controller_base*.whl."
        )

    flyte.init_from_config()
    run = flyte.run(fanout, n=fan_out)
    print(f"run name: {run.name}")
    print(f"run url:  {run.url}")
    run.wait()
