"""Ingest team (owner: Data Platform). Declares the data it owns and the task that cleans it.

The two handles at the top are this team's public interface. Other teams import them (they are plain
objects, so importing this module costs nothing) or restate them by name.

Deploy on its own:

    flyte deploy --root-dir . ingest/events.py env
"""

from __future__ import annotations

import io
from datetime import datetime

from cards import dataframe_card

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(
    name="ingest",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    labels={"team": "data-platform"},
)

# --- this team's public interface -------------------------------------------
# A handle names an artifact, its type, and its partition dimensions. Time
# dimensions carry a granularity so a range can be expanded into partitions.

raw_events = artifacts.Artifact(
    "raw_events",
    type=File,
    partitions={"date": artifacts.Daily, "region": str},
    description="Raw event CSVs as they land, one file per region per day.",
    source=True,  # nothing in the graph produces this; it lands from outside (see ingest/seed.py)
    kind="data",
)

events = artifacts.Artifact(
    "events",
    type=DataFrame,
    partitions={"date": artifacts.Daily, "region": str},
    description="Cleaned event stream, one partition per region per day.",
    kind="data",
).expect(region=["us", "eu"])  # expected values: a missing region is "never arrived", not "failed"
# ----------------------------------------------------------------------------


async def _parse(raw: File, min_quality: int):
    """Read a raw CSV and drop low-quality rows."""
    import pandas as pd

    async with raw.open("rb") as fh:
        data = await fh.read()
    df = pd.read_csv(io.BytesIO(bytes(data)))
    return df[df["quality"] >= min_quality].reset_index(drop=True)


@env.task(
    consumes_artifacts={
        "raw": raw_events,  # identity on date + region
        "date": raw_events.get_partition_value("date"),
        "region": raw_events.get_partition_value("region"),
    },
    produces_artifacts=(events,),
)
async def clean(raw: File, date: datetime, region: str, min_quality: int = 30) -> DataFrame:
    """Clean one region-day of raw events. Published as `events[date, region]`, with a data card."""
    df = await _parse(raw, min_quality)
    day = date.strftime("%Y-%m-%d")
    df["region"] = region
    df["date"] = day
    card = await artifacts.Card.create_from.aio(
        content=dataframe_card(
            df,
            name="events",
            description=f"Cleaned events for one region-day: rows with quality >= {min_quality} kept.",
            partition={"date": day, "region": region},
            task="ingest.clean",
            inputs=[f"raw_events[date={day}, region={region}]"],
        ),
        format="html",
        card_type="data",
    )
    return artifacts.new(DataFrame.from_df(df), events.at(date=date, region=region, card=card))
