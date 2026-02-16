import pathlib

import flyte


def build_runner_image() -> flyte.Image:
    image = flyte.Image.from_dockerfile(
        pathlib.Path(__file__).parent / "task_runner.dockerfile",
        registry="ghcr.io/flyteorg",
        name="n8n-task-runner-image",
    )
    return image


if __name__ == "__main__":
    flyte.init_from_config(image_builder="local")
    image = flyte.build(build_runner_image(), wait=True)
    print(image.uri)
