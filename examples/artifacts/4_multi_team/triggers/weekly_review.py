"""Pull on a timer, for an artifact you only read (owner: ML Quality, in its own repo).

ML Quality reviews the churn model every Monday morning and wants Sunday's model built by then, whether or not
anything else asked for it. It doesn't own `churn_model` (and can't import the ML team's module), so it states
it with `Artifact.ref` and keeps it fresh with `materialize_on`. That returns a generated environment whose one
task calls `flyte.materialize` for the partition at the trigger time minus `lag`; nobody writes that task.

Deploy records it as an inbound trigger on churn_model, so the graph draws it beside its target and
`flyte factory snapshot` turns it into `fc.on(flyte.Cron("0 5 * * 1"), churn_model, lag=TimeRange(days=1))`.
The owner's version of the same thing is `refresh=` on the handle (see `daily_report` in analytics/report.py).

    flyte deploy --root-dir . triggers/weekly_review.py for_weekly_review
"""

from __future__ import annotations

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})

# A failed materialization (a missing source partition, an unplannable graph) fails the run, so a broken
# Monday pull shows up as a failed run rather than a green one.
for_weekly_review = churn_model.materialize_on(
    flyte.Cron("0 5 * * 1"), lag=artifacts.TimeRange(days=1), name="for_weekly_review"
)
