"""
Submit N copies of the sleep_fanout `main` task as fast as possible.

Each run spawns n_children core-sleep leaves in leaseworker (no task pods).
Submissions are launched unbounded via asyncio.gather; we do not wait for
any run to finish — the harness returns as soon as all submissions are
accepted.
"""

import argparse
import asyncio
import time
from datetime import timedelta

import flyte

from sleep_fanout import main as sleep_fanout_main


async def submit_one(idx: int, n_children: int, sleep_duration: timedelta) -> str | None:
    try:
        run = await flyte.with_runcontext("remote").run.aio(
            sleep_fanout_main,
            n_children=n_children,
            sleep_duration=sleep_duration,
        )
        return run.url
    except Exception as e:
        print(f"[{idx}] submit failed: {e}", flush=True)
        return None


async def submit_many(total: int, n_children: int, sleep_duration: timedelta) -> None:
    start = time.monotonic()
    submitted = 0
    failed = 0

    async def wrapped(i: int):
        nonlocal submitted, failed
        name = await submit_one(i, n_children, sleep_duration)
        if name is None:
            failed += 1
        else:
            submitted += 1
            print(f"[{i}] {name}", flush=True)
        done = submitted + failed
        if done % 100 == 0:
            elapsed = time.monotonic() - start
            rps = done / elapsed if elapsed > 0 else 0
            print(f"submitted={submitted} failed={failed} elapsed={elapsed:.1f}s rps={rps:.1f}", flush=True)

    await asyncio.gather(*(wrapped(i) for i in range(total)))

    elapsed = time.monotonic() - start
    rps = submitted / elapsed if elapsed > 0 else 0
    print(f"\nDone. submitted={submitted} failed={failed} elapsed={elapsed:.2f}s rps={rps:.2f}")


# python stress/sleep_fanout_harness.py --total 10000 --n_children 10 --sleep_seconds 10
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2)
    parser.add_argument("--n_children", type=int, default=10)
    parser.add_argument("--sleep_seconds", type=int, default=10)
    args = parser.parse_args()

    flyte.init_from_config()
    asyncio.run(
        submit_many(
            total=args.total,
            n_children=args.n_children,
            sleep_duration=timedelta(seconds=args.sleep_seconds),
        )
    )


if __name__ == "__main__":
    main()
