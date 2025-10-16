"""Main entrypoint for the data pipeline with Flyte tasks.

This module defines Flyte tasks that orchestrate the business logic
from the data and models modules. It demonstrates:
- Async Flyte tasks
- Task chaining
- Integration with external dependencies
- Entrypoint pattern for execution
"""

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
    flyte.init_from_config()

    # Example API URL with mock data
    # In a real scenario, this would be a real API endpoint
    example_url = "https://jsonplaceholder.typicode.com/posts"

    # For demonstration, we'll use mock data instead of the actual API
    # to ensure the example works reliably
    print("Starting data pipeline...")
    print(f"Target API: {example_url}")

    # Create mock data for demonstration
    mock_data = {
        "items": [
            {"id": 1, "value": 10.5, "category": "Alpha"},
            {"id": 2, "value": 20.3, "category": "Beta"},
            {"id": 3, "value": 15.7, "category": "Alpha"},
            {"id": 4, "value": 8.9, "category": "Gamma"},
            {"id": 5, "value": 22.1, "category": "Beta"},
            {"id": 6, "value": 12.4, "category": "Alpha"},
            {"id": 7, "value": 18.6, "category": "Gamma"},
            {"id": 8, "value": 25.0, "category": "Beta"},
        ]
    }

    # To run remotely, uncomment the following:
    run = flyte.run(pipeline, api_url=example_url)
    print(f"\nRun Name: {run.name}")
    print(f"Run URL: {run.url}")
    run.wait()


if __name__ == "__main__":
    main()
