"""Main entrypoint for the data pipeline with Flyte tasks.

This module defines Flyte tasks that orchestrate the business logic
from the data and models modules. It demonstrates:
- Async Flyte tasks
- Task chaining
- Integration with external dependencies
- Entrypoint pattern for execution
"""
import pathlib
from typing import Any

import flyte

from pyproject_package.data import loader, processor
from pyproject_package.models import analyzer

# Define the task environment with resources and image
env = flyte.TaskEnvironment(
    name="data_pipeline",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=pathlib.Path("pyproject.toml")
    ),
    resources=flyte.Resources(memory="512Mi", cpu="500m"),
)


@env.task
async def fetch_task(url: str) -> dict[str, Any]:
    """Fetch data from an API endpoint.

    This task demonstrates async execution and external API calls.

    Args:
        url: API endpoint URL

    Returns:
        Raw data from the API
    """
    print(f"Fetching data from: {url}")
    data = await loader.fetch_data_from_api(url)
    print(f"Fetched {len(data)} top-level keys")
    return data


@env.task
async def process_task(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Process and transform raw data.

    This task demonstrates data cleaning and transformation.

    Args:
        raw_data: Raw data dictionary

    Returns:
        List of processed data items
    """
    print("Cleaning data...")
    cleaned = processor.clean_data(raw_data)

    print("Transforming data...")
    transformed = processor.transform_data(cleaned)

    print(f"Processed {len(transformed)} items")
    return transformed


@env.task
async def analyze_task(processed_data: list[dict[str, Any]]) -> str:
    """Analyze processed data and generate a report.

    This task demonstrates aggregation, statistical analysis, and reporting.

    Args:
        processed_data: List of processed data items

    Returns:
        Formatted analysis report
    """
    print("Aggregating data...")
    aggregated = await processor.aggregate_data(processed_data)

    print("Calculating statistics...")
    stats = analyzer.calculate_statistics(processed_data)

    print("Generating report...")
    report = analyzer.generate_report(
        {"basic": stats, "aggregated": aggregated}
    )

    print("\n" + report)
    return report


@env.task
async def pipeline(api_url: str) -> str:
    """Main data pipeline workflow.

    This task orchestrates the entire pipeline by chaining tasks together.

    Args:
        api_url: API endpoint to fetch data from

    Returns:
        Final analysis report
    """
    # Chain tasks together
    raw_data = await fetch_task(url=api_url)
    processed_data = await process_task(raw_data=raw_data)
    report = await analyze_task(processed_data=processed_data)

    return report


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
