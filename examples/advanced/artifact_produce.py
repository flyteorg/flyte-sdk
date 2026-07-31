"""Producer-side artifacts: declare produces_artifacts=True and wrap outputs with
flyte.artifacts.new(...). After the task succeeds, the platform extracts the stamped
artifact metadata from the outputs and records the generated artifacts on the action.
"""

import flyte
import flyte.artifacts as artifacts

env = flyte.TaskEnvironment(name="artifact_produce")


@env.task(produces_artifacts=True)
async def make_model() -> str:
    result = "model-weights-v1"
    metadata = artifacts.Metadata(
        name="my-produced-model",
        description="Produced by make_model",
        data={"framework": "torch"},
    )
    # No version in the metadata: the platform defaults it to <action_name>-<attempt>.
    return artifacts.new(result, metadata)


@env.task
async def make_plain() -> str:
    # No produces_artifacts flag and no wrapper: nothing is extracted.
    return "plain-output"


if __name__ == "__main__":
    flyte.init_from_config("/Users/ketanumare/src/flyte-sdk/.flyte/devbox.yaml")
    r = flyte.run(make_model)
    print("RUN", r.name)
    r.wait(quiet=True)
    print("OUTPUT", r.outputs())
