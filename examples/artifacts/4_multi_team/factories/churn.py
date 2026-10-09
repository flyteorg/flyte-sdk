"""Stage 5 of the journey: freeze the graph the teams grew into a factory you can run in production.

The emergent graph (stages 3-4) is always "whatever each team deployed last". That is what you want while
building, and exactly what you do not want behind a report the business reads every morning: one team's
redeploy should not silently change it. A factory is the same graph, written down once and versioned:

- every task is a deployed version, resolved when the factory deploys and pinned from then on;
- its triggers and sinks are part of it, so "what runs at 02:00 and who gets emailed" is reviewed in one file;
- `flyte factory diff churn` tells you what moved upstream since, and you adopt it by redeploying.

Nothing here re-implements a task. The factory reuses the teams' artifact handles (the same objects their
modules publish to) and calls their deployed tasks by name, so this file has no business logic at all.

    cd examples/artifacts/emergent_lineage
    flyte factory deploy factories/churn.py --dry-run     # validate against what is deployed, register nothing
    flyte factory deploy factories/churn.py
    flyte factory materialize churn send_report --partition date=2026-09-08 --wait

`churn.yaml` next to this file is the same factory as a spec: it is what `flyte factory snapshot` writes, and what
the console's factory editor reads and writes. Either one deploys; pick the one your reviewers prefer.
"""

from __future__ import annotations

import os

from analytics.report import daily_report
from flyteplugins.union import factory as fc
from ingest.events import events, raw_events
from ml.features import features
from ml.train import churn_model

import flyte
from flyte.remote import Task

# Each team's task, by its deployed name. No version: the factory resolves the latest one when it deploys and
# pins it, so the graph changes only when *you* redeploy this file (`flyte factory diff churn` shows what is new).
# Pin one explicitly with Task.get(name, version="...") to hold it back while the others move.


def _deployed(name: str):
    return Task.get(name, auto_version="latest")


# Optional: run the heavy step on its own queue (e.g. `testcluster` on a local devbox, a GPU queue in prod).
TRAIN_QUEUE = os.environ.get("CHURN_TRAIN_QUEUE")

# --- the graph: the same five handles the teams declared, wired once -----------------------------------------
# A handle passed to fc.source/fc.build brings its type, partitions, kind and description along. Inputs named
# like a partition dimension (`date`, `region`) receive the instance's value, exactly as in the emergent graph.

raw = fc.source(raw_events)
clean = fc.build(events).using(_deployed("ingest.clean"), raw=raw)
featurized = fc.build(features).using(_deployed("ml.featurize"), per_region=clean.all("region"))
model = fc.build(churn_model, runcontext={"queue": TRAIN_QUEUE} if TRAIN_QUEUE else None).using(
    _deployed("ml-train.train"), history=featurized.window(date=fc.TimeRange(days=30))
)
report = fc.build(daily_report).using(
    _deployed("analytics.report"), week=featurized.window(date=fc.TimeRange(days=7)), model=model
)

# A sink is a target that publishes nothing: materializing it builds the report it reads, then runs it.
report_sent = fc.sink("send_report").using(_deployed("analytics-notify.send_report"), report=report)

# An app is served from an artifact: deploying the factory points the deployed app at its model.
scoring = fc.serve("churn-scoring").using("churn-scoring", model=model)

churn = fc.Factory(
    "churn",
    report_sent,
    scoring,
    description="Daily churn report, emailed, plus the scoring app on the latest model.",
    triggers=[
        # Inbound: what daily_report's colocated `refresh=` policy did, now owned here.
        # Materializing the sink builds the report first, so one trigger covers both.
        fc.on(flyte.Cron("0 2 * * *"), report_sent, lag=fc.TimeRange(days=1), name="keep-report-fresh"),
        # Outbound: start the ML Quality team's check whenever this factory publishes a model. A name of its own:
        # deploy refuses to replace a same-named trigger the factory does not own (label factory=churn).
        fc.trigger(
            _deployed("ml-quality.validate"),
            on=model,
            inputs={"model": flyte.TriggeredArtifact, "threshold": 0.82},
            name="factory-revalidate-on-new-model",
        ),
    ],
)


if __name__ == "__main__":
    flyte.init_from_config()
    print(churn.graph())
