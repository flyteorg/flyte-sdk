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
def call_artifact() -> str:
    x = create_artifact()
    print(x)
    return x


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

    artifact_list_via_prefix = Artifact.list("my_artifact", partition_match="1.0")
    v4 = flyte.run(use_multiple_artifacts, v=artifact_list_via_prefix)
    print(v4.outputs())
