def normalize(values: list[float]) -> list[float]:
    """Normalize a list of values to [0, 1]."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def moving_average(values: list[float], window: int = 3) -> list[float]:
    """Compute a simple moving average."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result
