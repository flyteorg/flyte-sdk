"""Partitioned artifacts: publish a version per partition, then ask for one.

A partition is part of an artifact's identity, not a label. A version carries
at most one **time partition** (a date or datetime with a granularity) and any
number of **string partitions**, and the set of keys is fixed for the artifact
name by its first partitioned version:

    partitions={"date": date(2026, 8, 1), "region": "us"}

The value rule is the Python type you pass:

- a `date`     -> a daily time partition
- a `datetime` -> an hourly time partition (floored to the hour, UTC)
- `TimePartition(value, "week" | "month")` -> the coarser time granularities
- anything else -> a string partition (`str(value)`)

At most one time-valued entry is allowed. The key you use is the partition key,
so `{"date": ...}` names the time partition `date`.

What partitions buy you, all of which this example runs:

- `Artifact.get(name, date=..., region=...)` — the latest version of one partition
- `Artifact.listall(name, date=(lo, hi), latest_per_partition=True)` — one version
  per partition over a range, which is how a backfill plans its work
- `Artifact.partition_values(name, "region", date=...)` — the values a key has
- `Artifact.declare(name, {...})` — fix the keys before any version exists

Conformance is the registry's job. Publishing keys that disagree with the
artifact's schema does not fail: the version is stored and flagged, and
`Artifact.schema_mismatch` tells you so. A flagged version is not addressable
by partition, so it never answers a `get` or shows up in a range listing.

Try it:

    flyte run examples/artifacts/partitioned_artifacts.py main

or, to exercise the whole read side against a real registry:

    python examples/artifacts/partitioned_artifacts.py
"""

import asyncio
import tempfile
from datetime import date, datetime, timedelta, timezone

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="partitioned_artifacts")

REGIONS = ("us", "eu")


# 1. A task that produces one partitioned version per call. The partition values
#    are ordinary task inputs, so fanning the task out over (day, region) fills
#    one partition each.
@env.task(produces_artifacts=True)
async def ingest(day: date, region: str) -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{day} {region}: 42 events")
    file = await File.from_local(f.name)
    return artifacts.new(
        file,
        artifacts.Metadata(
            name="raw_events",
            # `day` is a date, so `date` is a daily time partition; `region` is
            # a string partition. These two keys are this artifact's identity.
            partitions={"date": day, "region": region},
            description=f"Raw events for {day} in {region}",
        ),
    )


# 2. An hourly artifact: pass a `datetime` and the time partition is hourly,
#    floored to the hour in UTC.
@env.task(produces_artifacts=True)
async def ingest_hourly(hour: datetime) -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{hour:%Y-%m-%dT%H}: 7 events")
    file = await File.from_local(f.name)
    return artifacts.new(file, artifacts.Metadata(name="raw_events_hourly", partitions={"hour": hour}))


# 3. A monthly rollup: `date` and `datetime` cover day and hour, so week and
#    month need TimePartition to say which one you mean.
@env.task(produces_artifacts=True)
async def roll_up(month: date) -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{month:%Y-%m}: 1260 events")
    file = await File.from_local(f.name)
    return artifacts.new(
        file,
        artifacts.Metadata(
            name="monthly_report",
            partitions={"date": artifacts.TimePartition(month, "month")},
        ),
    )


# 4. Consuming one partition: the artifact binds like any typed input. The
#    caller decides which partition to pass (see __main__ below).
@env.task
async def summarize(events: File) -> str:
    async with events.open("rb") as fh:
        return bytes(await fh.read()).decode()


# 5. A driver that fills a few partitions in one run.
@env.task
async def main(days: int = 3) -> list[str]:
    start = date(2026, 8, 1)
    coros = [ingest(start + timedelta(days=d), region) for d in range(days) for region in REGIONS]
    files = await asyncio.gather(*coros)
    await ingest_hourly(datetime(2026, 8, 1, 9, 30, tzinfo=timezone.utc))
    await roll_up(date(2026, 8, 1))
    return [f.path for f in files]


if __name__ == "__main__":
    from flyte.remote import Artifact

    flyte.init_from_config()

    # Declaring the keys before any version exists is optional. It is worth doing
    # when several producers write the same artifact, so the first one to publish
    # cannot fix the wrong keys by accident. `date` means daily, `str` a string
    # partition; "week" and "month" name the coarser time granularities.
    schema = Artifact.declare("raw_events", {"date": date, "region": str})
    print(f"raw_events keys: time={schema.time_key} ({schema.granularity}), strings={list(schema.keys)}")

    # Publishing from outside a task takes the same partitions= mapping.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("2026-08-04 us: 11 events (published from local)")
    published = Artifact.create(
        File.from_local_sync(f.name),
        name="raw_events",
        partitions={"date": date(2026, 8, 4), "region": "us"},
        python_type=File,
    )
    print(f"published {published.name}@{published.version} {published.partitions}")

    # Fill several partitions from tasks.
    run = flyte.run(main)
    print(run.url)
    run.wait()

    # -- reading ------------------------------------------------------------

    # The latest version of exactly one partition.
    latest = Artifact.get("raw_events", date=date(2026, 8, 1), region="us")
    print(f"one partition: {latest.version} {latest.partitions}")

    # One version per partition across a range: the (lo, hi) tuple on the time
    # key is an inclusive range, and latest_per_partition collapses each
    # partition to its newest version. This is the query a backfill plans with.
    august = list(
        Artifact.listall(
            "raw_events",
            date=(date(2026, 8, 1), date(2026, 8, 31)),
            latest_per_partition=True,
        )
    )
    print(f"august partitions: {len(august)}")
    for a in sorted(august, key=lambda a: (a.partitions["date"], a.partitions["region"])):
        print(f"  {a.partitions['date']} {a.partitions['region']:>2} -> {a.version}")

    # Narrow to one region by naming it: string keys take a value or a list.
    us_only = list(
        Artifact.listall(
            "raw_events",
            date=(date(2026, 8, 1), date(2026, 8, 31)),
            region="us",
            latest_per_partition=True,
        )
    )
    print(f"august us partitions: {len(us_only)}")

    # Without latest_per_partition the same query returns every version, not one
    # per partition: republishing a partition keeps the old version addressable
    # by its version id.
    every_version = list(Artifact.listall("raw_events", date=(date(2026, 8, 1), date(2026, 8, 31)), region="us"))
    print(f"august us versions: {len(every_version)}")

    # Which values does a key have? Optionally scoped by the other keys.
    print(f"regions: {Artifact.partition_values('raw_events', 'region')}")
    print(f"dates:   {Artifact.partition_values('raw_events', 'date', region='us')[:5]}")

    # Hourly and monthly artifacts read back as datetime and date respectively.
    hourly = Artifact.get("raw_events_hourly", hour=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc))
    print(f"hourly: {hourly.partitions}")
    monthly = Artifact.get("monthly_report", date=artifacts.TimePartition(date(2026, 8, 1), "month"))
    print(f"monthly: {monthly.partitions}")

    # Bind one partition's version to a task input.
    summary = flyte.run(summarize, events=Artifact.get("raw_events", date=date(2026, 8, 2), region="eu"))
    print(summary.url)

    # -- when the keys disagree ---------------------------------------------

    # raw_events is keyed by (date, region). Publishing (date, zone) does not
    # fail: the version is stored and flagged, and it is not addressable by
    # partition, so the get below finds nothing.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("wrong keys")
    flagged = Artifact.create(
        File.from_local_sync(f.name),
        name="raw_events",
        partitions={"date": date(2026, 8, 9), "zone": "a"},
        python_type=File,
    )
    print(f"flagged: {flagged.schema_mismatch}")
    try:
        Artifact.get("raw_events", date=date(2026, 8, 9), zone="a")
    except ValueError as e:
        print(f"as expected: {e}")
