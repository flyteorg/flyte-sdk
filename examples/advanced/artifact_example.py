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
def call_artifact(artifact: str) -> str:
    x = create_artifact()
    print(x)
    return artifact


if __name__ == "__main__":
    flyte.init()
    v = flyte.run(call_artifact, artifact="hello")
    print(v.outputs())
