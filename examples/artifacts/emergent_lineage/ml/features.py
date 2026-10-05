"""ML team (owner: ML Platform). Consumes across a dimension it does not own.

Featurization needs every region for one day, written as a mapping on its own input. The ingest team never
learns that featurization exists.

    flyte deploy --root-dir . ml/features.py env
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List

from cards import dataframe_card

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame

env = flyte.TaskEnvironment(
    name="ml",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    labels={"team": "ml"},
)

events = artifacts.Artifact.ref(
    "events",
    type=DataFrame,
    partitions={"date": artifacts.Daily, "region": str},
)

features = artifacts.Artifact(
    "features",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Model-ready feature table, one partition per day, all regions joined.",
    kind="data",
)


def _join(frames):
    """Per-user features across all regions of one day."""
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    out = df.groupby("user_id").agg(
        events=("event", "count"),
        buys=("event", lambda s: int((s == "buy").sum())),
        amount=("amount", "sum"),
    )
    out["churned"] = (out["buys"] == 0).astype(int)
    return out.reset_index()


@env.task(
    consumes_artifacts={"per_region": events.all("region")},
    # `date` needs no binding: a parameter with no default named like a dimension of `features` carries it.
    produces_artifacts=(features,),
)
async def featurize(per_region: List[DataFrame], date: datetime) -> DataFrame:
    """Join every region's events for `date` into one feature table, published with a data card."""
    import pandas as pd

    frames = await asyncio.gather(*(d.open(pd.DataFrame).all() for d in per_region))
    df = _join(frames)
    day = date.strftime("%Y-%m-%d")
    df["date"] = day
    card = await artifacts.Card.create_from.aio(
        content=dataframe_card(
            df,
            name="features",
            description=f"Per-user features for {day}, joined across {len(frames)} regions.",
            partition={"date": day},
            task="ml.featurize",
            inputs=[f"events[date={day}, region=*] ({len(frames)} partitions)"],
        ),
        format="html",
        card_type="data",
    )
    return artifacts.new(DataFrame.from_df(df), features.at(date=date, card=card))
