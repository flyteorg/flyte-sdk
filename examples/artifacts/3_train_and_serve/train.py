"""Train a model on a trailing week of data, and retrain whenever a new day of data is published.

    reviews[date] ──train (7-day window)──▶ sentiment_model[date]

Two declarations do all the work:

- `consumes_artifacts={"history": reviews.window(date=TimeRange(days=7))}`: building the model for a date reads
  the 7 days of reviews up to it.
- `refresh=artifacts.Refresh(reviews)` on the model: each new `reviews[date]` materializes
  `sentiment_model[date]`. Deploying this module registers that trigger; there is no scheduler to configure.

The task also acts as a quality gate: a model below `min_accuracy` on the newest day fails instead of being
published, so nothing downstream ever serves it.

    flyte deploy --root-dir . train.py env
"""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from datetime import datetime
from typing import List

from data import reviews

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(
    name="sentiment-train",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)

sentiment_model = artifacts.Artifact(
    "sentiment_model",
    type=File,
    partitions={"date": artifacts.Daily},
    kind="model",
    description="Word log-odds sentiment model trained on the 7 days of reviews up to `date`.",
    refresh=artifacts.Refresh(reviews, name="retrain_on_new_reviews"),  # each new day of data retrains
)


async def _rows(f: File) -> List[tuple]:
    async with f.open("rb") as fh:
        text = bytes(await fh.read()).decode()
    out = []
    for line in text.splitlines()[1:]:
        words, label = line.rsplit(",", 1)
        out.append((words.split(), int(label)))
    return out


def _score(model: dict, words: List[str]) -> float:
    return sum(model["weights"].get(w, 0.0) for w in words)


@env.task(
    consumes_artifacts={"history": reviews.window(date=artifacts.TimeRange(days=7))},
    produces_artifacts=(sentiment_model,),
)
async def train(history: List[File], date: datetime, min_accuracy: float = 0.75) -> File:
    """Fit word log-odds on all but the newest day; check accuracy on the newest day (the holdout)."""
    days = [await _rows(f) for f in history]
    fit, holdout = [r for d in days[:-1] for r in d], days[-1]
    pos, neg = Counter(), Counter()
    for words, label in fit:
        (pos if label else neg).update(words)
    vocab = set(pos) | set(neg)
    weights = {w: math.log((pos[w] + 1) / (neg[w] + 1)) for w in vocab}
    model = {"weights": weights, "trained_on": len(fit), "date": date.strftime("%Y-%m-%d")}
    model["accuracy"] = sum((_score(model, w) > 0) == bool(y) for w, y in holdout) / max(len(holdout), 1)
    if model["accuracy"] < min_accuracy:
        raise ValueError(f"accuracy {model['accuracy']:.2f} < {min_accuracy}: not publishing this model")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(model, f)
    return await File.from_local(f.name)  # published as sentiment_model[date]
