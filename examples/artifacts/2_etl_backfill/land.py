"""Stand-in for the export job that lands raw data from outside the pipeline.

It publishes `raw_trips[date, city]` at run time with `artifacts.new(...)`. Nothing is declared on the
decorator, so this task adds no edge to the lineage graph and `raw_trips` stays a source.

    flyte deploy --root-dir . land.py env
    flyte run --root-dir . land.py land --start 2026-09-01 --end 2026-09-14
    flyte run --root-dir . land.py land_trips --day 2026-09-15 --city nyc    # one file
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from datetime import date, timedelta
from typing import List

from pipeline import CITIES, raw_trips

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="trips-landing", image=flyte.Image.from_debian_base())


@env.task(produces_artifacts=True)
async def land_trips(day: date, city: str) -> File:
    """Write one day of fake trips for one city and publish it as `raw_trips[day, city]`."""
    rng = random.Random(f"{day}-{city}")
    rows = ["fare,km"] + [f"{rng.uniform(1, 60):.2f},{rng.uniform(0.5, 25):.1f}" for _ in range(50)]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("\n".join(rows) + "\n")
    return artifacts.new(await File.from_local(f.name), raw_trips.at(date=day, city=city))


@env.task
async def land(start: date, end: date, cities: List[str] = CITIES) -> int:
    """Land every city for every day in [start, end]. Returns the number of files."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    await asyncio.gather(*(land_trips(day=d, city=c) for d in days for c in cities))
    return len(days) * len(cities)
