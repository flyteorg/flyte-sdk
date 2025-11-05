"""
Custom type wrapper for positive integers.
This demonstrates how to create a custom type that wraps a built-in type.
"""


class PositiveInt:
    """A wrapper type that only accepts positive integers."""

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    def __int__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"PositiveInt({self._value})"

    def __str__(self) -> str:
        return str(self._value)

    def __eq__(self, other) -> bool:
        if isinstance(other, PositiveInt):
            return self._value == other._value
        return self._value == other

    def __hash__(self) -> int:
        return hash(self._value)
