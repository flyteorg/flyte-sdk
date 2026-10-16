"""A partitioned ETL pipeline, declared as artifacts instead of wired as a DAG.

    raw_trips[date, city] ──clean──▶ trips[date, city] ──summarize (all cities)──▶ daily_stats[date]
                                                                                       │
                                                       weekly_stats[date] ◀──rollup (7-day window)──┘

Every arrow is a task that names what it reads (`consumes_artifacts=`) and what it publishes
(`produces_artifacts=`). Nothing here calls another task, and there is no workflow function: ask for
`weekly_stats` on a date and the platform plans every partition it needs, reusing what is already built.

    flyte deploy --root-dir . pipeline.py env

See README.md for the walkthrough.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import List

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File

CITIES = ["nyc", "sf"]

env = flyte.TaskEnvironment(
    name="trips-etl",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)

# --- the artifacts -------------------------------------------------------------------------------------------
# A handle names an artifact, its type and its partition dimensions. `artifacts.Daily` makes `date` a time
# dimension, so a range like 2026-09-01..2026-09-07 expands into seven partitions.

raw_trips = artifacts.Artifact(
    "raw_trips",
    type=File,
    partitions={"date": artifacts.Daily, "city": str},
    description="Trip exports as they land: one CSV per city per day.",
    source=True,  # nothing in the graph builds it; it lands from outside (land.py stands in for that job)
)

trips = artifacts.Artifact(
    "trips",
    type=DataFrame,
    partitions={"date": artifacts.Daily, "city": str},
    description="Valid trips for one city and day.",
).expect(city=CITIES)  # the cities every day should have: a missing one is "never arrived", not "not built yet"

daily_stats = artifacts.Artifact(
    "daily_stats",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Trips, fares and distance per city for one day.",
    # Keep it fresh: every night at 03:00, build yesterday's stats (and whatever they need upstream).
    refresh=artifacts.Refresh(flyte.Cron("0 3 * * *"), lag=artifacts.TimeRange(days=1), name="nightly_stats"),
)

weekly_stats = artifacts.Artifact(
    "weekly_stats",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Trips and fares per city over the 7 days up to and including `date`.",
)
# -------------------------------------------------------------------------------------------------------------


@env.task(consumes_artifacts={"raw": raw_trips}, produces_artifacts=(trips,))
async def clean(raw: File, date: datetime, city: str, min_fare: float = 2.5) -> DataFrame:
    """Drop trips below `min_fare`.

    `raw` reads raw_trips for the same date and city (identity, the default). `date` and `city` have no
    binding: they are named like trips' dimensions, so each call receives the partition it builds.
    `min_fare` is a plain constant; changing it with `--input trips-etl.clean.min_fare=5` changes the cache key,
    so every partition it touches is rebuilt.
    """
    import pandas as pd

    async with raw.open("rb") as fh:
        df = pd.read_csv(io.BytesIO(bytes(await fh.read())))
    df = df[df["fare"] >= min_fare].reset_index(drop=True)
    df["date"], df["city"] = date.strftime("%Y-%m-%d"), city
    return DataFrame.from_df(df)  # published as trips[date, city]: the declaration says where


@env.task(consumes_artifacts={"per_city": trips.all("city")}, produces_artifacts=(daily_stats,))
async def summarize(per_city: List[DataFrame], date: datetime) -> DataFrame:
    """One row per city for one day. `trips.all("city")` fans in every city of the day being built."""
    import pandas as pd

    df = pd.concat([await d.open(pd.DataFrame).all() for d in per_city], ignore_index=True)
    out = df.groupby("city", as_index=False).agg(trips=("fare", "size"), fares=("fare", "sum"), km=("km", "sum"))
    out["date"] = date.strftime("%Y-%m-%d")
    return DataFrame.from_df(out)


@env.task(
    consumes_artifacts={"week": daily_stats.window(date=artifacts.TimeRange(days=7))},
    produces_artifacts=(weekly_stats,),
)
async def rollup(week: List[DataFrame], date: datetime) -> DataFrame:
    """Totals per city over the trailing week. The window is declared once, on the input that needs it."""
    import pandas as pd

    df = pd.concat([await d.open(pd.DataFrame).all() for d in week], ignore_index=True)
    out = df.groupby("city", as_index=False)[["trips", "fares", "km"]].sum()
    out["days"], out["date"] = len(week), date.strftime("%Y-%m-%d")
    return DataFrame.from_df(out)
