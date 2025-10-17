from typing import Tuple

import flyte
import flyte.artifacts as artifacts

env = flyte.TaskEnvironment("artifact_example")


@env.task
def create_artifact() -> str:
    result = "This is my artifact content"
    metadata = artifacts.Metadata(
        name="my_artifact", version="1.0", description="An example artifact created in create_artifact task"
    )
    return artifacts.new(result, metadata)


@env.task
def model_artifact() -> str:
    result = "This is my model artifact content"
    card = artifacts.Card.create_from(
        content="<h1>Model Card</h1><p>This is a sample model card.</p>",
        format="html",
        card_type="model",
    )

    metadata = artifacts.Metadata.create_model_metadata(
        name="my_model_artifact",
        version="1.0",
        description="An example model artifact created in model_artifact task",
        framework="PyTorch",
        model_type="Neural Network",
        architecture="ResNet50",
        task="Image Classification",
        modality=("image",),
        serial_format="pt",
        short_description="A ResNet50 model for image classification tasks.",
        card=card,
    )
    return artifacts.new(result, metadata)


@env.task
def call_artifact() -> Tuple[str, str]:
    x = create_artifact()
    print(x)
    y = model_artifact()
    print(y)
    return x, y


@env.task
async def use_artifact(v: str) -> str:
    print(f"Using artifact with content: {v}")
    return f"Artifact used with content: {v}"


@env.task
async def use_multiple_artifacts(v: list[str]) -> str:
    print(f"Using multiple artifacts with contents: {v}")
    return f"Multiple artifacts used with contents: {v}"


if __name__ == "__main__":
    flyte.init()
    v = flyte.run(call_artifact)
    print(v.outputs())

    from flyte.remote import Artifact

    artifact_instance = Artifact.get("my_artifact", version="1.0")
    v2 = flyte.run(use_artifact, v=artifact_instance)
    print(v2.outputs())

    artifact_list = [Artifact.get("my_artifact", version="1.0"), Artifact.get("my_artifact", version="1.0")]
    v3 = flyte.run(use_multiple_artifacts, v=artifact_list)
    print(v3.outputs())

    artifact_list_via_prefix = list(Artifact.listall("my_artifact", version="1.0"))
    v4 = flyte.run(use_multiple_artifacts, v=artifact_list_via_prefix)
    print(v4.outputs())
