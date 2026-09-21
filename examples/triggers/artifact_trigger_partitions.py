"""Artifact triggers scoped to a partition, with the partition bound to an input.

`OnArtifact` fires on every new version of an artifact. When the artifact is
partitioned, two things become available:

- **Narrow which versions fire.** `OnArtifact("raw_events", region="us")` fires
  only for versions whose `region` partition is `us`; versions for other regions
  are ignored. Only string partitions can narrow a trigger.
- **Bind the partition to an input.** `flyte.TriggeredPartition("date")` supplies
  the triggering version's `date` partition to a task input — a `datetime` for
  the time partition, a `str` for a string partition. It is the partition
  analogue of `flyte.TriggeredArtifact` (the version itself) and
  `flyte.TriggerTime` (a schedule's kickoff time).

So a downstream task can be told *which day and region just landed* without
opening the file or re-deriving it from the artifact's name.

A version whose partition keys do not match the artifact's schema is flagged by
the registry and never fires a trigger.

Try it:

    flyte deploy examples/triggers/artifact_trigger_partitions.py env
    flyte run examples/triggers/artifact_trigger_partitions.py producer --region us

then watch a run of `process_partition` appear, launched by the trigger with
`day` and `region` already filled in. Publishing with `--region eu` publishes a
version but fires nothing, because the trigger is scoped to `us`.
"""

import tempfile
from datetime import date, datetime

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="artifact_trigger_partitions_example")

process_us = flyte.Trigger(
    name="process-new-us-partition",
    # Only versions partitioned region=us fire this trigger.
    automation=flyte.OnArtifact(name="raw_events", region="us"),
    inputs={
        # The version itself.
        "events": flyte.TriggeredArtifact,
        # ...and its partition values, as ordinary typed inputs.
        "day": flyte.TriggeredPartition("date"),
        "region": flyte.TriggeredPartition("region"),
    },
    description="Process every new US partition of raw_events",
)


@env.task(produces_artifacts=True)
async def producer(day: date = date(2026, 8, 1), region: str = "us") -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{day} {region}: 42 events")
    file = await File.from_local(f.name)
    return artifacts.new(
        file,
        artifacts.Metadata(name="raw_events", partitions={"date": day, "region": region}),
    )


# A time partition arrives as a datetime (midnight UTC for a daily partition);
# a string partition arrives as a str.
@env.task(triggers=(process_us,))
async def process_partition(events: File, day: datetime, region: str) -> str:
    async with events.open("rb") as fh:
        content = bytes(await fh.read()).decode()
    result = f"processed {region} for {day:%Y-%m-%d}: {content}"
    print(result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Fires the trigger: region=us.
    run = flyte.run(producer, day=date(2026, 8, 1), region="us")
    print(f"published a us partition: {run.url}")

    # Publishes a version but fires nothing: the trigger is scoped to us.
    run = flyte.run(producer, day=date(2026, 8, 1), region="eu")
    print(f"published an eu partition (no trigger): {run.url}")
