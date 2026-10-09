"""Analytics team (owner: Analytics). Fan-in from two upstreams it does not own.

The Analytics team lives in its own repo, so it cannot import the ML team's handles. It states the two artifacts
it reads with `Artifact.ref`: their names and partitions, nothing the ML team decides (description, kind, ...).
`flyte deploy` checks both against the registry and prints the line to paste if ML has changed them since.

flyte deploy --root-dir . analytics/report.py env
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from typing import List

from cards import report_card

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(
    name="analytics",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    labels={"team": "analytics"},
)

# --- read from the ML team's repo: references, not imports -----------------------------------------
features = artifacts.Artifact.ref("features", type=DataFrame, partitions={"date": artifacts.Daily})
churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})

# --- this module's own artifact -----------------------------------------------------------------------
daily_report = artifacts.Artifact(
    "daily_report",
    type=File,
    partitions={"date": artifacts.Daily},
    description="Daily churn report: a week of features scored by the day's model.",
    kind="data",
    # Keep it fresh: every night at 02:00, materialize yesterday's report (and whatever it needs upstream).
    # Deploying this module registers the trigger; modules that only import daily_report do not.
    refresh=artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=artifacts.TimeRange(days=1), name="keep_report_fresh"),
)


async def _render(week: List[DataFrame], model: File, date: datetime):
    """The report as HTML, plus what its card shows: headline numbers and the users at risk."""
    import pandas as pd

    frames = [await d.open(pd.DataFrame).all() for d in week]
    df = pd.concat(frames, ignore_index=True)
    async with model.open("rb") as fh:
        weights = json.loads(bytes(await fh.read()))
    churned = weights["churned"]
    buys = df.groupby("user_id")["buys"].sum()
    at_risk = buys[buys <= churned.get("buys", 0)].sort_values()
    users = df["user_id"].nunique()
    html = (
        f"<h1>Churn report {date:%Y-%m-%d}</h1>"
        f"<p>{len(week)} days of features, {users} users, {len(at_risk)} at risk.</p>"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
    stats = [("days of features", len(week)), ("users", users), ("at risk", len(at_risk))]
    return await File.from_local(f.name), html, stats, [(u, int(b)) for u, b in at_risk.head(20).items()]


@env.task(
    consumes_artifacts={
        "week": features.window(date=artifacts.TimeRange(days=7)),
        "model": churn_model,  # identity on date
        "date": artifacts.partition("date"),  # the date being built; same as daily_report.get_partition_value
    },
    produces_artifacts=(daily_report,),
)
async def report(week: List[DataFrame], model: File, date: datetime) -> File:
    """Render `daily_report[date]`, with a card showing the report and who is at risk.

    Returning the file alone would publish it from the declaration alone; `artifacts.new(...)` is only
    needed here to attach the card.
    """
    file, html, stats, at_risk = await _render(week, model, date)
    day = date.strftime("%Y-%m-%d")
    card = await artifacts.Card.create_from.aio(
        content=report_card(
            html,
            name="daily_report",
            description="Daily churn report: a week of features scored by the day's model.",
            partition={"date": day},
            stats=stats,
            task="analytics.report",
            inputs=[f"features[7 days to {day}]", f"churn_model[date={day}]"],
            at_risk=at_risk,
        ),
        format="html",
        card_type="data",
    )
    return artifacts.new(file, daily_report.at(date=date, card=card))
