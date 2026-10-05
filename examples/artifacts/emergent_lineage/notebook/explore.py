"""The researcher never touches any of this (proposal section 7).

The handles are importable, so the same names work from a notebook, and the level-0 path still works for
someone who declares nothing.

    python notebook/explore.py --config <config.yaml> --date 2026-09-08
"""

from __future__ import annotations

import argparse
from datetime import datetime

from ml.features import features

import flyte
from flyte.io import DataFrame
from flyte.remote import Artifact


def main(day: datetime) -> None:
    # Typed fetch through the handle: the date is floored to the handle's Daily granularity.
    by_handle = Artifact.get(features, date=day)
    print("features via handle:", by_handle.name, by_handle.version)

    # Or by name, with no import at all.
    by_name = Artifact.get("features", date=day.date())
    print("features via name:  ", by_name.name, by_name.version)

    # Browse what exists.
    for a in Artifact.listall(name="churn_model", limit=5):
        print("churn_model version:", a.version)

    # A range through the handle, one version per partition.
    month = list(
        Artifact.listall(
            features, date=flyte.TimeRange(f"{day:%Y-%m}-01", f"{day:%Y-%m-%d}"), latest_per_partition=True
        )
    )
    print(f"features this month: {len(month)} partitions")

    # Publish an experiment from the notebook. No handle, no declaration, no deploy.
    import pandas as pd

    exp = Artifact.create(
        DataFrame.from_df(pd.DataFrame({"user_id": ["u1"], "score": [0.42]})),
        name="churn_features_exp42",
        python_type=DataFrame,
    )
    print("published", exp.name, exp.version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--date", default="2026-09-08")
    args = parser.parse_args()
    flyte.init_from_config(args.config)
    main(datetime.fromisoformat(args.date))
