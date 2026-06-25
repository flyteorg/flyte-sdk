"""
End-to-end test for `run_base_dir` (ENG-26-673).

`run_base_dir` is a *full* storage path used verbatim — bucket and all — as the base
under which a run's metadata (inputs.pb, per-action outputs, action metadata) is
written. The backend appends per-run namespacing under it, so you supply the base,
not the full per-action path. Pointing it at a different bucket than the backend's
configured default should "just work" (cross-bucket requires the dataplane storage
client to allow dynamic container loading / enable-multicontainer).

What this exercises:
  - dataproxy.UploadInputs(base_dir=...)  -> writes the root action's inputs.pb
  - RunSpec.run_base_dir                  -> CreateRun resolves the run output base
  - a parent task that fans out to a child task, so you can see the per-action
    namespacing (a0 = root, plus child actions) land under your chosen base.

Uses .flyte/config.yaml from this repo. Override with FLYTE_CONFIG=<path> to target a
different backend.

    # default base (control): metadata goes under the backend default
    uv run python examples/run_base_dir_e2e.py

    # explicit base in the *default* bucket, different prefix
    uv run python examples/run_base_dir_e2e.py s3://<default-bucket>/eng26673-test

    # explicit base in a *different* bucket (true bucket switch)
    uv run python examples/run_base_dir_e2e.py s3://<other-bucket>/eng26673-test

The script prints the run URL and the expected storage prefix. After the run, verify
the objects landed under <run_base_dir>/... and NOT under the backend's default
location:

    aws s3 ls --recursive <run_base_dir>/

On the cloud path the root inputs land at
    <run_base_dir>/<org>/<project>/<domain>/<run>/a0/inputs.pb
with per-action outputs.pb under the same run output base.
"""

import os
import sys
from pathlib import Path

import flyte

# Config file, resolved relative to this repo so it works regardless of cwd.
# Override with FLYTE_CONFIG=<path> to target a different backend.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / ".flyte" / "config.yaml"
CONFIG_PATH = Path(os.environ["FLYTE_CONFIG"]) if os.environ.get("FLYTE_CONFIG") else _DEFAULT_CONFIG

env = flyte.TaskEnvironment(name="run_base_dir_e2e")


@env.task
async def square(x: int) -> int:
    return x * x


@env.task
async def sum_of_squares(n: int = 5) -> int:
    """Parent action (a0). Fans out to `square` so child actions also write under the base."""
    import asyncio

    with flyte.group("squares"):
        results = await asyncio.gather(*(square(i) for i in range(n)))
    total = sum(results)
    print(f"sum of squares 0..{n - 1} = {total}")
    return total


def main() -> None:
    # First positional arg overrides run_base_dir; omit it to use the backend default.
    run_base_dir = sys.argv[1] if len(sys.argv) > 1 else None

    flyte.init_from_config(str(CONFIG_PATH))

    ctx = flyte.with_runcontext(run_base_dir=run_base_dir) if run_base_dir else flyte.with_runcontext()
    run = ctx.run(sum_of_squares, n=5)

    print("\n=== run_base_dir e2e ===")
    print(f"run_base_dir (requested): {run_base_dir or '(backend default)'}")
    print(f"run name: {run.name}")
    print(f"run url:  {run.url}")
    if run_base_dir:
        base = run_base_dir.rstrip("/")
        print("\nExpected metadata location (cloud layout):")
        print(f"  inputs (root):  {base}/<org>/<project>/<domain>/{run.name}/a0/inputs.pb")
        print(f"  outputs (root): {base}/<org>/<project>/<domain>/{run.name}/a0/outputs.pb")
        print("\nVerify it landed there (and NOT under the backend default):")
        print(f"  aws s3 ls --recursive {base}/")
    print("========================\n")


if __name__ == "__main__":
    main()
