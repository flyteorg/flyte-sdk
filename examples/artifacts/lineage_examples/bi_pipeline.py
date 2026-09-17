"""Business intelligence use case: ingest -> two aggregation levels -> report, fan-in.

A fan-in lineage: `aggregate_weekly` extends the chain from `aggregate_daily`,
but `build_report` then consumes *both* `daily-aggregates` and
`weekly-aggregates` -- a run with two upstream artifacts at different chain
depths, rather than the ETL example's two-independent-roots diamond.
`bi_dashboard_app` serves the report and, like the other two examples, needs
the `upstream-artifact-name`/`upstream-artifact-version` labels since
it fetches the report by name rather than binding it as a typed input.

Try it:

    python examples/artifacts/lineage_examples/bi_pipeline.py
"""

import csv
import io
import json
import tempfile
import time
from collections import defaultdict

import fastapi
from lineage import LABEL_UPSTREAM_ARTIFACT_NAME, LABEL_UPSTREAM_ARTIFACT_VERSION

import flyte
import flyte.artifacts as artifacts
from flyte.app.extras import FastAPIAppEnvironment
from flyte.io import File

# Shared by the task env and the dashboard app below: the module imports `fastapi` at
# top level (for `bi_app`), so any task defined here needs it too, even ones that never
# touch FastAPI themselves -- task loading imports the whole module.
image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")
env = flyte.TaskEnvironment(name="bi_pipeline", image=image)


def _wait_for_artifact(name: str, *, retries: int = 15, delay_s: float = 2.0):
    """`Artifact.get` right after a run completes can race artifact indexing; retry briefly."""
    from flyte.remote import Artifact

    for attempt in range(retries):
        try:
            return Artifact.get(name)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def _run_and_wait(run):
    """`.wait()` alone doesn't raise on failure -- surface it immediately instead of
    letting a later `_wait_for_artifact` call fail with a confusing "not found"."""
    run.wait()
    if run.phase != "succeeded":
        raise RuntimeError(f"{run.url} finished in phase {run.phase!r}")
    return run


_EVENTS = [
    ("2024-01-01", "1", "12.50"),
    ("2024-01-01", "2", "4.00"),
    ("2024-01-02", "3", "9.25"),
    ("2024-01-08", "4", "30.00"),
    ("2024-01-08", "5", "5.75"),
]


@env.task(produces_artifacts=True)
async def ingest_events() -> File:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["day", "event_id", "revenue"])
    writer.writerows(_EVENTS)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="raw-events", description="Raw revenue events"))


@env.task(produces_artifacts=True)
async def aggregate_daily(events: File) -> File:
    async with events.open("rb") as fh:
        rows = list(csv.DictReader(io.StringIO(bytes(await fh.read()).decode())))
    by_day: dict[str, float] = defaultdict(float)
    for row in rows:
        by_day[row["day"]] += float(row["revenue"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(by_day), f)
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="daily-aggregates", description="Revenue summed per day"))


@env.task(produces_artifacts=True)
async def aggregate_weekly(daily: File) -> File:
    async with daily.open("rb") as fh:
        by_day = json.loads(bytes(await fh.read()).decode())
    # ISO week number from the date string, no datetime parsing needed for this toy range.
    week_of = {"2024-01-01": "w1", "2024-01-02": "w1", "2024-01-08": "w2"}
    by_week: dict[str, float] = defaultdict(float)
    for day, revenue in by_day.items():
        by_week[week_of[day]] += revenue
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(by_week), f)
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="weekly-aggregates", description="Revenue summed per week"))


# Fan-in: consumes both the daily and weekly aggregates, at different chain depths.
# Both edges are found via the automatic bound-input scan.
@env.task(produces_artifacts=True)
async def build_report(daily: File, weekly: File) -> File:
    async with daily.open("rb") as fh:
        by_day = json.loads(bytes(await fh.read()).decode())
    async with weekly.open("rb") as fh:
        by_week = json.loads(bytes(await fh.read()).decode())
    report = {"total_revenue": sum(by_day.values()), "by_day": by_day, "by_week": by_week}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(report, f)
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="bi-report", description="Revenue report"))


bi_app = fastapi.FastAPI(title="BI Dashboard")


@bi_app.get("/report")
async def report() -> dict:
    import json as _json

    from flyte.io import File as _File
    from flyte.remote import Artifact

    report_artifact = await Artifact.get.aio(name="bi-report")
    report_file = await report_artifact.to_python(_File)
    async with report_file.open("rb") as fh:
        return _json.loads(bytes(await fh.read()).decode())


# Labels are set on this module-level object in `run_pipeline` below, once the report's
# identity is known -- not via `clone_with()`, which would return a local-variable copy
# the app resolver can't find by name in this module.
bi_dashboard_app = FastAPIAppEnvironment(name="bi-dashboard-app", app=bi_app, image=image)


def run_pipeline() -> None:
    """Ingest, aggregate at two levels, build the report, and serve it. Assumes init_from_config() ran."""
    ingest_run = _run_and_wait(flyte.run(ingest_events))
    print(ingest_run.url)
    events = _wait_for_artifact("raw-events")

    daily_run = _run_and_wait(flyte.run(aggregate_daily, events=events))
    print(daily_run.url)
    daily = _wait_for_artifact("daily-aggregates")

    weekly_run = _run_and_wait(flyte.run(aggregate_weekly, daily=daily))
    print(weekly_run.url)
    weekly = _wait_for_artifact("weekly-aggregates")

    report_run = _run_and_wait(flyte.run(build_report, daily=daily, weekly=weekly))
    print(report_run.url)
    report_artifact = _wait_for_artifact("bi-report")

    bi_dashboard_app.labels = {
        LABEL_UPSTREAM_ARTIFACT_NAME: report_artifact.name,
        LABEL_UPSTREAM_ARTIFACT_VERSION: report_artifact.version,
    }
    flyte.serve(bi_dashboard_app)


if __name__ == "__main__":
    flyte.init_from_config()
    run_pipeline()
