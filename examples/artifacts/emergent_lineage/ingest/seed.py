"""Seed raw data (owner: Data Platform). Stands in for whatever lands `raw_events` from outside.

It publishes with the ladder's lowest rungs: `produces_artifacts=True` plus `artifacts.new(file,
raw_events.at(...))` at run time. No handle is declared on the decorator, so this task adds no edge to the
graph and `raw_events` stays a source node.

    flyte deploy --root-dir . ingest/seed.py env
    flyte run --root-dir . ingest/seed.py seed --start 2026-08-01 --end 2026-09-08
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from datetime import date, timedelta
from typing import List

from cards import csv_card

import flyte
import flyte.artifacts as artifacts
from flyte.io import File
from ingest.events import raw_events

env = flyte.TaskEnvironment(
    name="ingest-seed",
    image=flyte.Image.from_debian_base(),
    labels={"team": "data-platform"},
)

REGIONS = ["us", "eu"]


def _fake_csv(day: date, region: str, rows: int = 20) -> str:
    rng = random.Random(f"{day}-{region}")
    lines = ["user_id,event,quality,amount"]
    for i in range(rows):
        lines.append(f"u{rng.randint(1, 8)},{rng.choice(['view', 'click', 'buy'])},{rng.randint(0, 100)},{i % 7}")
    return "\n".join(lines) + "\n"


@env.task(produces_artifacts=True)
async def publish_raw(day: date, region: str) -> File:
    """Write one tiny CSV and publish it as `raw_events[day, region]`, with a card showing its rows."""
    text = _fake_csv(day, region)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(text)
    file = await File.from_local(f.name)
    card = await artifacts.Card.create_from.aio(
        content=csv_card(
            text,
            name="raw_events",
            description="Raw event CSV as it landed: one file per region per day.",
            partition={"date": day.isoformat(), "region": region},
            task="ingest-seed.publish_raw",
        ),
        format="html",
        card_type="data",
    )
    return artifacts.new(file, raw_events.at(date=day, region=region, card=card))


@env.task
async def seed(start: date, end: date, regions: List[str] = REGIONS) -> int:
    """Publish `raw_events` for every day in [start, end] and every region. Returns the count."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    await asyncio.gather(*(publish_raw(day=d, region=r) for d in days for r in regions))
    return len(days) * len(regions)
