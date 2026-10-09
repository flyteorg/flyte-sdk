"""Production: every new day of data retrains the model and redeploys the app serving it, as one versioned unit.

The live graph (data.py, train.py, serve.py) already retrains on new data, through `refresh=` on the model.
Serving the new model is a separate `flyte materialize app` call. A factory puts both behind one trigger:

    reviews ──on new version──▶ train (7-day window) ──▶ sentiment_model ──serve──▶ sentiment-api

`fc.on(reviews)` materializes everything this factory produces downstream of `reviews`: the model for the
day that landed, then the app pinned to it. The task versions are pinned when the factory deploys, so a
teammate redeploying `train.py` changes nothing here until you redeploy this file (`flyte factory diff
sentiment` shows what moved).

Use one or the other for a given model: when this factory owns retraining, drop `refresh=` from
`sentiment_model` in train.py, or each new day trains twice (the second is a cache hit, but it still runs).

Experimental: needs the Union lineage service and flyteplugins-union.

    flyte factory deploy factory.py --dry-run       # check it against what is deployed; registers nothing
    flyte factory deploy factory.py
    flyte factory materialize sentiment sentiment-api --partition date=2026-09-14 --wait
"""

from __future__ import annotations

from data import reviews
from flyteplugins.union import factory as fc
from train import sentiment_model

import flyte
from flyte.remote import Task

source = fc.source(reviews)
model = fc.build(sentiment_model).using(
    Task.get("sentiment-train.train", auto_version="latest"),  # the deployed task; pinned when this deploys
    history=source.window(date=fc.TimeRange(days=7)),
)
api = fc.serve("sentiment-api").using("sentiment-api", model=model)  # the deployed app, pointed at `model`

sentiment = fc.Factory(
    "sentiment",
    api,
    description="Retrain on each new day of reviews and serve the new model.",
    triggers=[fc.on(source, name="retrain-and-serve")],
)


if __name__ == "__main__":
    flyte.init_from_config()
    print(sentiment.graph())
