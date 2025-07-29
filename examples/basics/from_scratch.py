import flyte

env = flyte.TaskEnvironment(
    name="squarer",
    resources=flyte.Resources(
        cpu=1,
    ),
)


@env.task
def square(x: int) -> int:
    return x * x


@env.task
def main(n: int) -> list[int]:
    """
    Calculate the square of numbers from 0 to n-1.

    Args:
        n: The upper limit (exclusive) for the range of numbers to square.

    Returns:
        A list of squared numbers.
    """
    return list(flyte.map(square, range(n)))


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(main, n=10)
    print(run.url)
