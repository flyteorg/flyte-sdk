"""Caller-declared artifacts: publish another task's outputs without changing it.

`artifacts.new(...)` works when you own the task and can wrap its return value.
When you don't (a shared library task, a remote task reference, anything written
before artifacts existed), the caller declares which outputs are artifacts:

    with artifacts.produces(o0=artifacts.Metadata(name="clean_events", partitions={...})):
        await clean.override(produces_artifacts=True)(day=day, region=region)

- Keyword names are output slots: `o0` is the first output, `o1` the second.
  Slots you leave out are ordinary outputs.
- The called task must run with `produces_artifacts=True`; that flag is what lets
  the platform publish its outputs.
- The artifact is published from the called action's outputs, only when it
  succeeds, and its source is that action, exactly as if the task had returned
  `artifacts.new(...)` itself.
- Call one task per block. The declaration reaches that task only, never the
  actions it spawns, and it is hidden from `flyte.get_custom_context()`.
- If the task also wraps the output itself, the caller's declaration wins for the
  name, version, partitions and parents; the task's description and attrs fill in
  whatever the caller left empty.

Try it:

    python examples/artifacts/produced_artifacts.py
"""

import asyncio
import tempfile
from datetime import date, timedelta

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="produced_artifacts")

REGIONS = ("us", "eu")


# 1. Tasks that know nothing about artifacts. They return plain values.
@env.task
async def clean(day: date, region: str) -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{day} {region}: 40 clean events")
    return await File.from_local(f.name)


@env.task
async def train(features: File) -> tuple[File, File]:
    """Returns (model, training log). Only the model is worth publishing."""
    with tempfile.NamedTemporaryFile("w", suffix=".pt", delete=False) as m:
        m.write(f"weights trained on {features.path}")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        log.write("epoch 1 loss 0.42")
    return await File.from_local(m.name), await File.from_local(log.name)


# 2. The caller decides what gets published, and under which name and partition.
async def clean_one(day: date, region: str) -> File:
    with artifacts.produces(o0=artifacts.Metadata(name="clean_events", partitions={"date": day, "region": region})):
        return await clean.override(produces_artifacts=True)(day=day, region=region)


@env.task
async def main(days: int = 2) -> str:
    start = date(2026, 8, 1)
    # One declaration per call, so concurrent calls keep their own partitions.
    files = await asyncio.gather(*(clean_one(start + timedelta(days=d), r) for d in range(days) for r in REGIONS))

    # Declare only the first output. `o1`, the training log, stays an ordinary
    # output and is not published.
    with artifacts.produces(o0=artifacts.Metadata(name="events_model", kind="model", description="Daily model")):
        model, _log = await train.override(produces_artifacts=True)(features=files[0])
    return model.path


if __name__ == "__main__":
    from flyte.remote import Artifact

    flyte.init_from_config()

    run = flyte.run(main)
    print(run.url)
    run.wait()

    # The declared outputs are ordinary artifacts, partitioned as declared.
    for a in Artifact.listall("clean_events", date=(date(2026, 8, 1), date(2026, 8, 2)), latest_per_partition=True):
        print(f"clean_events {a.partitions} -> {a.version}")

    model = Artifact.get("events_model")
    print(f"events_model {model.version} kind={model.kind}")

    # The same works for a task you only have a reference to. Look it up, turn on
    # publishing for this call, and declare its outputs:
    #
    #     ref = flyte.remote.Task.get("other_team.clean", auto_version="latest")
    #     callee = await ref.override.aio(produces_artifacts=True)
    #     with artifacts.produces(o0=artifacts.Metadata(name="clean_events", partitions={...})):
    #         await callee(day=day, region=region)
