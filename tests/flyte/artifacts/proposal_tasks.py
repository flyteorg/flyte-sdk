"""The proposal's four tasks plus the section 8 adapter, as declared there (bodies are stubs)."""

from datetime import datetime
from typing import List

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File

ingest = flyte.TaskEnvironment(name="ingest")
ml = flyte.TaskEnvironment(name="ml")
analytics = flyte.TaskEnvironment(name="analytics")

raw_events = artifacts.Artifact(
    "raw_events", type=File, partitions={"date": artifacts.Daily, "region": str}, source=True
)
events = artifacts.Artifact(
    "events",
    type=DataFrame,
    partitions={"date": artifacts.Daily, "region": str},
    description="Cleaned event stream, one partition per region per day.",
)
features = artifacts.Artifact(
    "features",
    type=DataFrame,
    partitions={"date": artifacts.Daily},
    description="Model-ready feature table, one partition per day, all regions joined.",
)
churn_model = artifacts.Artifact("churn_model", type=File, partitions={"date": artifacts.Daily}, kind="model")
daily_report = artifacts.Artifact("daily_report", type=File, partitions={"date": artifacts.Daily})


@ingest.task(
    produces_artifacts=(events,),
    consumes_artifacts={
        "raw": raw_events,
        "date": raw_events.get_partition_value("date"),
        "region": raw_events.get_partition_value("region"),
    },
)
async def clean(raw: File, date: datetime, region: str, min_quality: int = 30) -> DataFrame:
    raise NotImplementedError


@ml.task(
    produces_artifacts=(features,),
    consumes_artifacts={"per_region": events.all("region"), "date": features.get_partition_value("date")},
)
async def featurize(per_region: list[DataFrame], date: datetime) -> DataFrame:
    raise NotImplementedError


@ml.task(
    produces_artifacts=(churn_model,),
    consumes_artifacts={
        "history": features.window(date=artifacts.TimeRange(days=30)),
        "date": churn_model.get_partition_value("date"),
    },
)
async def train(history: list[DataFrame], date: datetime, lr: float = 3e-4) -> File:
    raise NotImplementedError


@analytics.task(
    produces_artifacts=(daily_report,),
    consumes_artifacts={
        "week": features.window(date=artifacts.TimeRange(days=7)),
        "model": churn_model,
        "date": daily_report.get_partition_value("date"),
    },
)
async def report(
    week: List[DataFrame],
    model: File,
    date: datetime,
) -> File:
    raise NotImplementedError


@ingest.task(
    produces_artifacts=(events,),
    consumes_artifacts={
        "raw": raw_events,
        "date": raw_events.get_partition_value("date"),
        "region": raw_events.get_partition_value("region"),
    },
)
async def clean_adapter(raw: File, date: datetime, region: str) -> DataFrame:
    raise NotImplementedError


ALL = [clean, featurize, train, report]
