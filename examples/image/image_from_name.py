import logging

import flyte

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

noop_env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_name("custom-image")
    .with_apt_packages("ca-certificates")
    .with_pip_packages("flyte", pre=True)
    .with_local_v2(),
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
    #     custom-image: debian:stable
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
