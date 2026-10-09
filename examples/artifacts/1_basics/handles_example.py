"""Stage 2 of the journey: declare artifacts beside the code that makes them.

Stage 1 (`artifact_example.py`, `partitioned_artifacts.py`) publishes artifacts by name at run time. That
already beats passing S3 paths around, but nothing knows ahead of time what a task makes or reads: the
dependency between `clean_orders` and `daily_revenue` lives only in someone's head.

Here one team declares two artifact *handles* at the top of its module, and names them in the task
decorators. That is the whole change, and it buys three things before anything runs:

- Deploy checks the declarations against the signatures (a dimension nothing supplies, an input no handle
  feeds, a typo in a name) and refuses to ship a pipeline that cannot be planned.
- Deploy records what each task produces and consumes, so the console draws the graph
  (`raw_orders → clean_orders → orders → daily_revenue → revenue`) without anyone drawing it.
- The platform can now *pull*: ask for `revenue` on a date and it plans every upstream task for the
  partitions that date needs (`../2_etl_backfill/`).

    flyte deploy handles_example.py env          # prints "3 tasks, 3 artifact handles, 2 dependency edges resolved"
    flyte run handles_example.py publish_raw --day 2026-09-08
    flyte materialize artifact revenue --partition date=2026-09-08 --plan

Next: `../2_etl_backfill/` builds and backfills a partitioned pipeline from handles like these, and
`../4_multi_team/` is the same idea across four teams and thirteen modules.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import List

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(
    name="orders",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    labels={"team": "commerce"},
)

# --- this module's public interface --------------------------------------------------------------
# A handle is a contract: a name, a type, and the partition dimensions every version carries. Other
# modules import these objects (they are plain values) instead of restating strings.

raw_orders = artifacts.Artifact(
    "raw_orders",
    type=File,
    partitions={"date": artifacts.Daily},
    description="Order exports as they land, one CSV per day.",
    source=True,  # published from outside the graph (publish_raw stands in for the export job)
    kind="data",
)

orders = artifacts.Artifact(
    "orders",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Valid orders for one day.",
    kind="data",
)

revenue = artifacts.Artifact(
    "revenue",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Revenue per product over the trailing 3 days, as of one day.",
    kind="data",
)
# --------------------------------------------------------------------------------------------------


@env.task(produces_artifacts=True)
async def publish_raw(day: date) -> File:
    """Stand-in for the export job: publish one day of raw orders (the lowest rung: no declaration)."""
    import tempfile

    rows = "order_id,product,amount,status\n" + "".join(
        f"{i},{['tea', 'mug', 'pot'][i % 3]},{5 + i % 7},{'ok' if i % 9 else 'void'}\n" for i in range(40)
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(rows)
    return artifacts.new(await File.from_local(f.name), raw_orders.at(date=day))


@env.task(consumes_artifacts={"raw": raw_orders}, produces_artifacts=(orders,))
async def clean_orders(raw: File, date: datetime) -> DataFrame:
    """Drop voided orders. `date` has no binding: it is named like orders' dimension, so it receives it."""
    import pandas as pd

    async with raw.open("rb") as fh:
        df = pd.read_csv(io.BytesIO(bytes(await fh.read())))
    df = df[df["status"] == "ok"].reset_index(drop=True)
    df["date"] = date.strftime("%Y-%m-%d")
    return DataFrame.from_df(df)


@env.task(
    consumes_artifacts={"trailing_data": orders.window(date=artifacts.TimeRange(days=3))},  # the 3 days up to `date`
    produces_artifacts=(revenue,),
)
async def daily_revenue(trailing_data: List[DataFrame], date: datetime) -> DataFrame:
    """Revenue per product over the trailing 3 days. The window is declared once, on the input that needs it."""
    import pandas as pd

    frames = [await d.open(pd.DataFrame).all() for d in trailing_data]
    df = pd.concat(frames, ignore_index=True).groupby("product", as_index=False)["amount"].sum()
    df["date"] = date.strftime("%Y-%m-%d")
    return DataFrame.from_df(df)
