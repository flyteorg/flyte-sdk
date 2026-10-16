"""Stage 4 of the journey: pull what you need, and let the graph work out the rest.

Once the teams' declarations are deployed (stage 3), nobody writes an orchestration DAG to get a report. You
ask for the artifact partition you want; the platform walks the declarations backwards, reuses every
partition that already has a version for the same code and inputs, and builds only what is missing.

    python pull.py --config <flyte config> plan      # what would run for 2026-09-08, nothing launched
    python pull.py --config <flyte config> day       # build daily_report for one day
    python pull.py --config <flyte config> backfill  # a week, 8 days in flight at once
    python pull.py --config <flyte config> rebuild   # re-featurize after a fix the cache cannot see
    python pull.py --config <flyte config> send      # run the send_report sink (and whatever it needs)
    python pull.py --config <flyte config> app       # build what churn-scoring reads, then redeploy it serving that

The same calls exist on the CLI (`flyte materialize artifact daily_report --partition date=2026-09-08 --plan`,
`flyte materialize app churn-scoring --partition date=2026-09-08`), in the
console (the play button on any node of the lineage graph), and on a schedule: `refresh=` on the handle for the
owner (`daily_report` in analytics/report.py), `handle.materialize_on(...)` for anyone else
(`triggers/weekly_review.py`).
"""

from __future__ import annotations

import argparse
from datetime import datetime

from analytics.notify import send_report
from analytics.report import daily_report
from apps.scoring import scoring
from ml.features import features

import flyte
import flyte.artifacts as artifacts

DAY = datetime(2026, 9, 8)


def plan() -> None:
    """Plan only: the instance DAG and the cache probe, with nothing launched."""
    result = flyte.materialize(daily_report, date=DAY, plan_only=True)
    print(result)


def day() -> None:
    """One partition. Everything upstream that is already built for the same task versions is reused."""
    run = flyte.materialize(daily_report, date=DAY)
    print(run.url)
    run.wait()


def backfill() -> None:
    """A range is the same walk over more partitions; `concurrency` bounds how many build at once."""
    week = artifacts.TimeRange(datetime(2026, 9, 2), datetime(2026, 9, 8))
    run = flyte.materialize(daily_report, date=week, concurrency=8)
    print(run.url)
    run.wait()


def rebuild() -> None:
    """Force one step (and everything downstream of it) to run again, e.g. after fixing a bug in a dependency
    the cache key cannot see. Steps upstream of it are still reused."""
    run = flyte.materialize(daily_report, date=DAY, rebuild=["ml.featurize"])
    print(run.url)
    run.wait()


def send() -> None:
    """A sink is a target too: send_report publishes nothing, so materializing it plans the report it reads and
    then runs it (never from cache: running it is the point)."""
    run = flyte.materialize(send_report, date=DAY)
    print(run.url)
    run.wait()


def app() -> None:
    """An app is a target too: materializing churn-scoring builds the churn_model its `model` parameter reads
    (and whatever that needs), then redeploys the app serving the version the run built."""
    run = flyte.materialize(scoring, date=DAY)
    print(run.url)
    run.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="flyte config file (default: the usual lookup)")
    parser.add_argument("what", choices=["plan", "day", "backfill", "rebuild", "send", "app"])
    args = parser.parse_args()
    flyte.init_from_config(args.config)
    {"plan": plan, "day": day, "backfill": backfill, "rebuild": rebuild, "send": send, "app": app}[args.what]()
    _ = features  # imported so the handles above resolve the same way the deployed modules do
