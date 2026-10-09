"""The ETL walkthrough from Python: plan, build, backfill, change a parameter, and find missing data.

Each step is one `flyte.materialize(...)` call. You never say which tasks to run: you ask for partitions of
`weekly_stats`, and the planner walks pipeline.py's declarations backwards to `raw_trips`, reusing every
partition already built by the same code from the same inputs.

    python backfill.py --config <flyte config> plan        # what one day needs; launches nothing
    python backfill.py --config <flyte config> day         # build weekly_stats for 2026-09-07
    python backfill.py --config <flyte config> backfill    # 2026-09-08 .. 2026-09-14, 8 builds at a time
    python backfill.py --config <flyte config> again       # the same range: everything is reused
    python backfill.py --config <flyte config> min-fare    # a new `clean.min_fare`: rebuilds what depends on it
    python backfill.py --config <flyte config> missing     # a day whose raw files never landed

Experimental: needs the Union lineage service and flyteplugins-union.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from pipeline import weekly_stats

import flyte
import flyte.artifacts as artifacts

DAY = datetime(2026, 9, 7)
WEEK = artifacts.TimeRange(datetime(2026, 9, 8), datetime(2026, 9, 14))


def plan() -> None:
    """The instance DAG for one day: 7 days of daily_stats, 14 trips partitions, 14 raw files to read."""
    print(flyte.materialize(weekly_stats, date=DAY, plan_only=True))


def day() -> None:
    run = flyte.materialize(weekly_stats, date=DAY)
    print(run.url)
    run.wait()


def backfill() -> None:
    """A range is the same walk over more partitions. The days it shares with `day` are reused, not rebuilt."""
    run = flyte.materialize(weekly_stats, date=WEEK, concurrency=8)
    print(run.url)
    run.wait()


def again() -> None:
    """Rerunning a backfill is safe and cheap: every partition is a cache hit."""
    backfill()


def min_fare() -> None:
    """A constant is part of the cache key. Overriding it rebuilds clean and everything downstream of it for
    the requested partitions, and only those; the raw files are read as they are."""
    run = flyte.materialize(weekly_stats, date=DAY, inputs={"trips-etl.clean.min_fare": 5.0})
    print(run.url)
    run.wait()


def missing() -> None:
    """Nothing landed for 2026-09-20: the plan names the raw_trips partitions it would need, before anything
    runs. Land them (`flyte run --root-dir . land.py land --start 2026-09-14 --end 2026-09-20`) and try again."""
    try:
        flyte.materialize(weekly_stats, date=datetime(2026, 9, 20), plan_only=True)
    except flyte.errors.MaterializeError as e:
        print(e)


if __name__ == "__main__":
    steps = {"plan": plan, "day": day, "backfill": backfill, "again": again, "min-fare": min_fare, "missing": missing}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="flyte config file (default: the usual lookup)")
    parser.add_argument("step", choices=list(steps))
    args = parser.parse_args()
    flyte.init_from_config(args.config)
    steps[args.step]()
