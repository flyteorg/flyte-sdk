"""Tests for sandboxed source extraction and return rewriting."""

import pytest

from flyte.sandbox._source import extract_source


def test_simple_function_body():
    def add(x: int, y: int) -> int:
        return x + y

    code, input_names = extract_source(add)
    assert input_names == ["x", "y"]
    assert "__result__" in code
    assert "x + y" in code


def test_multiline_function():
    def compute(a: int, b: int) -> int:
        c = a + b
        d = c * 2
        return d

    code, input_names = extract_source(compute)
    assert input_names == ["a", "b"]
    assert "__result__ = d" in code
    assert "c = a + b" in code
    assert "d = c * 2" in code


def test_no_return():
    def noop(x: int) -> None:
        _ = x + 1

    code, input_names = extract_source(noop)
    assert input_names == ["x"]
    # No return statement means only the trailing __result__ expression (no assignment)
    assert "__result__ =" not in code
    assert code.endswith("__result__")


def test_bare_return():
    def bare(x: int) -> None:
        if x > 0:
            return

    code, input_names = extract_source(bare)
    assert input_names == ["x"]
    assert "__result__ = None" in code


def test_multiple_returns():
    def branching(x: int) -> int:
        if x > 0:
            return x
        return -x

    code, input_names = extract_source(branching)
    assert input_names == ["x"]
    # 2 assignments + 1 trailing expression
    assert code.count("__result__") == 3


def test_rejects_async_function():
    async def bad(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="cannot be async"):
        extract_source(bad)


def test_rejects_generator():
    def bad(x: int):
        yield x

    with pytest.raises(TypeError, match="cannot be generators"):
        extract_source(bad)


def test_rejects_async_generator():
    async def bad(x: int):
        yield x

    with pytest.raises(TypeError, match="cannot be async generators"):
        extract_source(bad)


def test_preserves_conditionals():
    def classify(x: int) -> str:
        if x > 0:
            return "positive"
        elif x < 0:
            return "negative"
        else:
            return "zero"

    code, input_names = extract_source(classify)
    assert input_names == ["x"]
    # 3 assignments + 1 trailing expression
    assert code.count("__result__") == 4


def test_string_operations():
    def greet(name: str) -> str:
        return f"hello {name}"

    code, input_names = extract_source(greet)
    assert input_names == ["name"]
    assert "__result__" in code
