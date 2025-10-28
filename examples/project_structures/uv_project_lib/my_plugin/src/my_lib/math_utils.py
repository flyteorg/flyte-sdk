"""Mathematical utility functions."""


def linear_function(x: int, slope: int = 2, intercept: int = 5) -> int:
    """Apply a linear function: y = slope * x + intercept."""
    return slope * x + intercept


def calculate_mean(values: list[int]) -> float:
    """Calculate the arithmetic mean of a list of values."""
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)
