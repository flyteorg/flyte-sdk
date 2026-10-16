"""The training data: one partition of labeled reviews per day, published as it lands.

`reviews` is a source artifact: nothing in the graph builds it. In real life a labeling job or an export
publishes it; here `land_reviews` stands in for that job. Publishing a new day is the event the rest of this
example reacts to.

    flyte deploy --root-dir . data.py env
    flyte run --root-dir . data.py land --start 2026-09-01 --end 2026-09-14
    flyte run --root-dir . data.py land_reviews --day 2026-09-15        # one more day: retrains the model
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from datetime import date, timedelta

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

reviews = artifacts.Artifact(
    "reviews",
    type=File,
    partitions={"date": artifacts.Daily},
    description="Labeled product reviews (text, label), one CSV per day.",
    source=True,
)

env = flyte.TaskEnvironment(name="reviews-landing", image=flyte.Image.from_debian_base())

_GOOD = ["great", "love", "fast", "works", "perfect", "sturdy", "recommend"]
_BAD = ["broken", "slow", "refund", "awful", "late", "flimsy", "disappointed"]
_FILLER = ["the", "it", "this", "box", "item", "was", "and", "really", "very"]


@env.task(produces_artifacts=True)
async def land_reviews(day: date, rows: int = 200) -> File:
    """Write one day of fake labeled reviews and publish it as `reviews[day]`."""
    rng = random.Random(str(day))
    lines = ["text,label"]
    for _ in range(rows):
        label = rng.random() < 0.5
        words = rng.sample(_GOOD if label else _BAD, 2) + rng.sample(_FILLER, 4)
        rng.shuffle(words)
        if rng.random() < 0.12:  # mislabeled rows, as real labels have: accuracy lands near 0.88, not 1.0
            label = not label
        lines.append(f"{' '.join(words)},{int(label)}")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    return artifacts.new(await File.from_local(f.name), reviews.at(date=day))


@env.task
async def land(start: date, end: date) -> int:
    """Land every day in [start, end]. Returns the number of days."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    await asyncio.gather(*(land_reviews(day=d) for d in days))
    return len(days)
