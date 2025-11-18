"""
Simple Poetry Project Example

This example demonstrates how to use Poetry for dependency management with Flyte.
Poetry is a Python dependency management and packaging tool that uses pyproject.toml
and poetry.lock files to ensure reproducible builds.
"""

import pathlib

import flyte

# Create a task environment that uses Poetry for dependency management
env = flyte.TaskEnvironment(
    name="poetry_example",
    resources=flyte.Resources(memory="250Mi"),
    image=(
        flyte.Image.from_debian_base().with_poetry_project(
            pyproject_file=pathlib.Path("pyproject.toml"),
            poetry_lock=pathlib.Path("poetry.lock"),
            project_install_mode="dependencies_only",
            # Optional: pass extra args to poetry install
            # ,extra_args="--no-root",  # Use this if you don't want to install the project itself
        )
        .with_env_vars({"foo": "baeeewdsddeeeeddddddddreeedsddeeeeeeeeeddeereeddreyreteeeeewdde"})
    ),
)


@env.task
def calculate(x: int) -> int:
    """Simple calculation task."""
    import numpy as np

    # Use numpy (installed via poetry)
    slope, intercept = 2, 5
    result = slope * x + intercept
    print(f"Calculated: {slope} * {x} + {intercept} = {result}")
    return result


@env.task
def fetch_data(url: str) -> str:
    """Fetch data from a URL using requests (installed via poetry)."""
    import requests

    response = requests.get(url)
    print(f"Fetched from {url}, status: {response.status_code}")
    return f"Status: {response.status_code}, Length: {len(response.content)}"


@env.task
def main(x_list: list[int]) -> dict:
    """Main workflow that processes a list of numbers."""
    print("hi")
    import numpy as np

    x_len = len(x_list)
    if x_len < 5:
        raise ValueError(f"x_list needs at least 5 items, found: {x_len}")

    # Run calculations in parallel using flyte.map
    results = list(flyte.map(calculate, x_list))

    # Calculate statistics using numpy
    y_mean = float(np.mean(results))
    y_std = float(np.std(results))

    # Fetch some example data
    data_info = fetch_data("https://httpbin.org/get")

    return {
        "mean": y_mean,
        "std": y_std,
        "results": results,
        "data_info": data_info,
    }


if __name__ == "__main__":
    # Initialize remote connection
    flyte.init_from_config()

    # Run the workflow remotely
    run = flyte.run(main, x_list=list(range(10)))

    # Print run information
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")

    # Wait for completion and stream logs
    run.wait(run)

