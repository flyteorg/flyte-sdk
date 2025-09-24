import pathlib

import flyte
from my_custom_lib import linear_function, calculate_mean

env = flyte.TaskEnvironment(
    name="uv_project_lib_tasks",
    resources=flyte.Resources(memory="250Mi"),
    image=(
        flyte.Image.from_debian_base().with_uv_project(
            pyproject_file=pathlib.Path("my_custom_lib/pyproject.toml"),  # SDK should not add this pyproject to code bundle by default.
            pre=True,
        ).with_local_v2()
    ),
)


@env.task
def process_value(x: int) -> int:
    """Process a single value using the custom linear function."""
    return linear_function(x)


@env.task
def process_list(x_list: list[int]) -> float:
    """Process a list of values and return the mean of processed values."""
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a large enough sample size, found: {x_len}")

    # flyte.map is like Python map, but runs in parallel.
    y_list = list(flyte.map(process_value, x_list))
    y_mean = calculate_mean(y_list)
    print(f"Processed {x_len} values, mean result: {y_mean}")
    return y_mean


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(process_list, x_list=list(range(10)))

    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
