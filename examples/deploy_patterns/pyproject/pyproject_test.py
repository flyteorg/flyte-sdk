import pathlib

import flyte

env = flyte.TaskEnvironment(
    name="pyproject_test_0",
    resources=flyte.Resources(memory="250Mi"),
    image=(
        flyte.Image.from_debian_base(registry="docker.io/nielsbantilan").with_uv_project(
            pyproject_file=pathlib.Path("pyproject.toml"),
            pre=True,
        )
    ),
)


@env.task
def fn(x: int) -> int:  # Type annotations are recommended.
    slope, intercept = 2, 5
    return slope * x + intercept


@env.task
def main(x_list: list[int]) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a larger enough sample size, found: {x_len}")

    # flyte.map is like Python map, but runs in parallel.
    y_list = list(flyte.map(fn, x_list))
    y_mean = sum(y_list) / len(y_list)
    print(y_mean)
    return y_mean


if __name__ == "__main__":
    # Establish a remote connection from within your script.
    flyte.init_from_config("../../../config.yaml")

    # Run your tasks remotely inline and pass parameter data.
    run = flyte.run(main, x_list=list(range(10)))

    # Print various attributes of the run.
    print(run.name)
    print(run.url)

    # Stream the logs from the remote run to the terminal.
    run.wait(run)
