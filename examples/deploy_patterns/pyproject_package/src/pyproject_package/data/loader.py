"""Data loading utilities.

This module provides functions to load data from various sources,
including APIs and local files. It has no Flyte dependencies and
can be used independently.
"""
import json
from pathlib import Path
from typing import Any

import httpx


async def fetch_data_from_api(url: str) -> list[dict[str, Any]]:
    """Fetch data from an API endpoint asynchronously.

    Args:
        url: The API endpoint URL

    Returns:
        Dictionary containing the API response data

    Example:
        >>> import asyncio
        >>> data = asyncio.run(fetch_data_from_api("https://api.example.com/data"))
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()


def load_local_data(file_path: str | Path) -> dict[str, Any]:
    """Load data from a local JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the file data

    Example:
        >>> data = load_local_data("data.json")
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r") as f:
        return json.load(f)
