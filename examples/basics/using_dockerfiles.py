from pathlib import Path

import flyte
from flyte.extras import ContainerTask

env = flyte.TaskEnvironment(
    name="dockerfile_example",
    image=flyte.Image.from_dockerfile(
        file=Path(__file__).parent / "Dockerfile.example",
        name="flyte-temp-dockerfile-example",
        registry="ghcr.io/flyteorg",
        platform=("linux/amd64", "linux/arm64"),
    ),
)


version_task = ContainerTask(
    image=env.image,
    name="example_task",
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs={},
    outputs={"greeting": str},
    command=["/bin/sh", "-c", "python --version | tee -a /var/outputs/version"],
)

env.add_task(version_task)


if __name__ == "__main__":
    flyte.init_from_config("../config.yaml")
    print(flyte.run(version_task))
