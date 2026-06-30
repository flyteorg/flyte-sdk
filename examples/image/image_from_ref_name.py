import logging

import flyte

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

noop_env = flyte.TaskEnvironment(
    name="env_from_image_ref_name",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_ref_name("custom-image"),
)


@noop_env.task
async def main(x: int = 1) -> int:
    logger.debug(f"Task noop: {x}")
    return x


if __name__ == "__main__":
    # The image references can be set in config like following:
    #
    # image:
    #   image_refs:
    #     custom-image: ghcr.io/flyteorg/flyte:py3.13-v2.0.0b52
    flyte.init_from_config(images=("custom-image=ghcr.io/flyteorg/flyte:py3.13-v2.0.0b52",))
    run = flyte.run(main)
    print(run.name)
    print(run.url)
