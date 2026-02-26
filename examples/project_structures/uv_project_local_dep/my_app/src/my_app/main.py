import logging
import pathlib

import flyte
from my_lib.math_utils import calculate_mean, linear_function

UV_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent  # my_app

env = flyte.TaskEnvironment(
    name="uv_project_local_dep",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=UV_PROJECT_ROOT / "pyproject.toml",
    ),
)


@env.task
def process_value(x: int) -> int:
    """Process a single value using the linear function from my_lib."""
    return linear_function(x)


@env.task
def process_list(x_list: list[int]) -> float:
    """Process a list of values and return the mean of processed values."""
    x_len = len(x_list)
    y_list = list(flyte.map(process_value, x_list))
    y_mean = calculate_mean(y_list)
    print(f"Processed {x_len} values, mean result: {y_mean}")
    return y_mean


if __name__ == "__main__":
    flyte.init_from_config(root_dir=UV_PROJECT_ROOT.parent, log_level=logging.DEBUG)

    run = flyte.run(process_list, x_list=list(range(3)))

    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
