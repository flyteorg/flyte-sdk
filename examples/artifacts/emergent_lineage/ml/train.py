"""ML team (owner: ML Platform). A trailing window, and publishing at run time.

The handle is declared on the decorator, so the node and its edges exist at deploy; the value is still
published with `artifacts.new(...)` at run time because the model card is rendered from metrics that only
exist once the fit has finished. That run-time metadata wins over the declaration's defaults.

    flyte deploy --root-dir . ml/train.py env
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from typing import List, Tuple

from cards import model_card

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File
from ml.features import features

env = flyte.TaskEnvironment(
    name="ml-train",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    labels={"team": "ml"},
)

churn_model = artifacts.Artifact(
    "churn_model",
    type=File,
    partitions={"date": artifacts.Daily},
    kind="model",
    description="Churn classifier trained on a 30-day feature window.",
)


async def _fit(history: List[DataFrame], lr: float) -> Tuple[File, dict, dict]:
    """A deliberately tiny 'model': per-feature means of churned vs retained users."""
    import pandas as pd

    frames = [await d.open(pd.DataFrame).all() for d in history]
    df = pd.concat(frames, ignore_index=True)
    cols = ["events", "buys", "amount"]
    weights = {
        "churned": df[df["churned"] == 1][cols].mean().fillna(0).to_dict(),
        "retained": df[df["churned"] == 0][cols].mean().fillna(0).to_dict(),
        "lr": lr,
    }
    metrics = {"rows": len(df), "days": len(history), "churn_rate": float(df["churned"].mean() if len(df) else 0)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(weights, f)
    return await File.from_local(f.name), weights, metrics


@env.task(
    consumes_artifacts={
        "history": features.window(date=artifacts.TimeRange(days=30)),
        "date": churn_model.get_partition_value("date"),  # not features: that window spans 30 of them
    },
    produces_artifacts=(churn_model,),
)
async def train(history: List[DataFrame], date: datetime, lr: float = 3e-4) -> File:
    """Fit on 30 days of features; publish `churn_model[date]` with a model card."""
    model, weights, metrics = await _fit(history, lr=lr)
    day = date.strftime("%Y-%m-%d")
    card = await artifacts.Card.create_from.aio(
        content=model_card(
            weights,
            metrics,
            name="churn_model",
            description="Churn classifier: per-feature means of churned vs retained users over 30 days.",
            partition={"date": day},
            task="ml-train.train",
            inputs=[f"features[date in the 30 days to {day}] ({len(history)} partitions)"],
        ),
        format="html",
        card_type="model",
    )
    return artifacts.new(model, churn_model.at(date=date, card=card))
