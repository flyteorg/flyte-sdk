"""Utility functions for workflows."""


def process_data(data: str) -> str:
    """Process data."""
    return f"Processed: {data}"


def validate_input(value: int) -> bool:
    """Validate input."""
    return value > 0
