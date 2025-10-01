import logging

import flyte

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

noop_env = flyte.TaskEnvironment(
    name="reuse_concurrency",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=100,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_name("custom-image"),
)


@noop_env.task
async def main(x: int) -> int:
    logger.debug(f"Task noop: {x}")
    return x


if __name__ == "__main__":
    from click.testing import CliRunner

    from flyte.cli.main import main as cli_main

    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        [
            "run",
            "--image",
            # Assign image uri to name custom-image
            "custom-image=ghcr.io/flyteorg/flyte",
            "examples/image/image_from_name.py",
            "main",
            "--x",
            "42",
        ],
    )

    print("Exit code:", result.exit_code)
    print("Output:", result.output)
