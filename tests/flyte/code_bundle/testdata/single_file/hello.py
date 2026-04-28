def fn(x: int) -> int:
    slope, intercept = 2, 5
    return slope * x + intercept


def main(values: list[int]) -> float:
    total = sum(fn(v) for v in values)
    return total / len(values)
