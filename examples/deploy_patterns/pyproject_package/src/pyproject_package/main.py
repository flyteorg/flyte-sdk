"""Main entrypoint for the data pipeline with Flyte tasks.

This module defines Flyte tasks that orchestrate the business logic
from the data and models modules. It demonstrates:
- Async Flyte tasks
- Task chaining
- Integration with external dependencies
- Entrypoint pattern for execution
"""

import pathlib

import flyte
from pyproject_package.tasks.tasks import pipeline


def main():
    """Main entry point for the pipeline.

    This function can be called from:
    - The installed script: `run-pipeline`
    - As a module: `python -m pyproject_package.main`
    - Directly: `python src/pyproject_package/main.py`
    """
    # Initialize Flyte connection
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent.parent)

    # Example API URL with mock data
    # In a real scenario, this would be a real API endpoint
    example_url = "https://jsonplaceholder.typicode.com/posts"

    # For demonstration, we'll use mock data instead of the actual API
    # to ensure the example works reliably
    print("Starting data pipeline...")
    print(f"Target API: {example_url}")

    # To run remotely, uncomment the following:
    run = flyte.run(pipeline, api_url=example_url)
    print(f"\nRun Name: {run.name}")
    print(f"Run URL: {run.url}")
    run.wait()


if __name__ == "__main__":
    main()
