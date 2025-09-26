import pathlib

import flyte
from my_lib.math_utils import calculate_mean, linear_function

env = flyte.TaskEnvironment(
    name="uv_project_lib_task_in_src",
    resources=flyte.Resources(memory="1000Mi"),
    image=(flyte.Image.from_debian_base()),
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
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent.parent
    )  # TODO: SDK should fail early if root dir is None

    run = flyte.run(process_list, x_list=list(range(10)))

    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
