"""Artifact triggers: fire a run when a new artifact version is created.

The `consumer` task deploys with an `OnArtifact` trigger watching the
`customer_model` artifact. Every time `producer` runs and publishes a new
version, the platform fires a run of `consumer`, binding the fresh artifact to
the `model` input (the `flyte.TriggeredArtifact` sentinel — the artifact-trigger
analogue of `flyte.TriggerTime`) and filling `threshold` from the trigger's
default inputs.

Try it:

    flyte deploy examples/triggers/artifact_trigger.py env
    flyte run examples/triggers/artifact_trigger.py producer --version v1

then watch a run of `consumer` appear, launched by the trigger.
"""

import tempfile

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="artifact_trigger_example")

retrain = flyte.Trigger(
    name="retrain-on-new-model",
    automation=flyte.OnArtifact(name="customer_model"),  # any new version fires
    inputs={"model": flyte.TriggeredArtifact, "threshold": 0.5},
    description="Validate every new customer_model version",
)


@env.task(produces_artifacts=True)
async def producer(version: str = "v1") -> File:
    with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as f:
        f.write(f"model weights {version}")
    file = await File.from_local(f.name)
    return artifacts.new(
        file,
        artifacts.Metadata(name="customer_model", version=version, description="Demo model"),
    )


@env.task(triggers=[retrain])
async def consumer(model: File, threshold: float = 0.5) -> str:
    async with model.open("rb") as fh:
        content = bytes(await fh.read()).decode()
    result = f"validated {content!r} at threshold {threshold}"
    print(result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(producer, version="v1")
    print(run.url)
