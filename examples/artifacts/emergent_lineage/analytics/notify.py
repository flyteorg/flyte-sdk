"""A sink: send each day's report out (owner: Analytics).

`send_report` consumes `daily_report` (identity) and publishes nothing, so deploy records it as a sink: a step at
the end of the graph that `flyte factory snapshot` compiles to `fc.sink(...)`. Its `date` parameter has no
default and no binding, and is named like `daily_report`'s dimension, so it carries the date of the report it
receives (the implicit rule; `artifacts.partition("date")` says the same thing explicitly).

    flyte deploy --root-dir . analytics/notify.py env
"""

from __future__ import annotations

import logging
from datetime import datetime

import flyte
from analytics.report import daily_report
from flyte.io import File

env = flyte.TaskEnvironment(
    name="analytics-notify",
    image=flyte.Image.from_debian_base(),
    labels={"team": "analytics"},
)

log = logging.getLogger(__name__)


@env.task(consumes_artifacts={"report": daily_report})
async def send_report(report: File, date: datetime, to: str = "analytics@example.com") -> str:
    """'Send' the day's report: log where it would go. Returns a receipt; nothing is published."""
    async with report.open("rb") as fh:
        size = len(bytes(await fh.read()))
    receipt = f"sent daily_report[{date:%Y-%m-%d}] ({size} bytes) to {to}"
    log.info(receipt)
    return receipt
