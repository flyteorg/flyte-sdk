"""Artifact triggers fired by external publishes: CLI or `Artifact.create`.

`OnArtifact` does not care how an artifact version is born. The sibling
example (`artifact_trigger.py`) fires on task-produced artifacts; this one
shows the other two ways — no producer task anywhere:

- CLI: `flyte create artifact ...` uploads a local file and publishes it as a
  File artifact.
- Programmatic: `flyte.remote.Artifact.create(...)` from any Python process —
  a script, a notebook, an external system's callback.

Either way the platform fires a run of `process_incoming_data`, binding the fresh artifact
to the `dataset` input via the `flyte.TriggeredArtifact` sentinel.

Try it:

    flyte deploy examples/triggers/artifact_trigger_external.py env

then publish a version from the CLI:

    echo "id,value" > /tmp/data.csv
    flyte create artifact incoming_dataset --from-file /tmp/data.csv --version v1

or programmatically (also what `__main__` below does):

    python examples/triggers/artifact_trigger_external.py

and watch a run of `process_incoming_data` appear, launched by the trigger.
"""

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="artifact_trigger_external_example")

ingest = flyte.Trigger(
    name="process-on-new-dataset",
    automation=flyte.OnArtifact(name="incoming_dataset"),  # any new version fires
    inputs={"dataset": flyte.TriggeredArtifact},
    description="Process every incoming_dataset version, however it was published",
)


@env.task(triggers=(ingest,))
async def process_incoming_data(dataset: File) -> str:
    async with dataset.open("rb") as fh:
        content = bytes(await fh.read()).decode()
    result = f"processed {len(content)} bytes: {content[:50]!r}"
    print(result)
    return result


if __name__ == "__main__":
    import tempfile

    from flyte.remote import Artifact

    flyte.init_from_config()

    # Programmatic publish from this process — no task, no run. The upload and
    # the artifact record both happen inside Artifact.create.
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,value\n1,a\n2,b\n")
    published = Artifact.create(
        File.from_local_sync(f.name),
        name="incoming_dataset",
        description="Dataset dropped off by an external process",
        external_ref="s3://partner-bucket/drop/2026-08-06.csv",  # provenance for out-of-band data
    )
    print(f"published {published.name}@{published.version} — the trigger fires a run of `process_incoming_data`")
    print(f"view it at {published.url}")
