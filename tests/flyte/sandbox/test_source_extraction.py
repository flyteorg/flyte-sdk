"""Tests for sandboxed source extraction."""

import pytest

from flyte.sandbox._source import extract_source


def test_simple_function_body():
    def add(x: int, y: int) -> int:
        return x + y

    code, input_names = extract_source(add)
    assert input_names == ["x", "y"]
    assert "def add" in code
    assert "return x + y" in code
    assert code.endswith("add(x, y)")


def test_multiline_function():
    def compute(a: int, b: int) -> int:
        c = a + b
        d = c * 2
        return d

    code, input_names = extract_source(compute)
    assert input_names == ["a", "b"]
    assert "def compute" in code
    assert "c = a + b" in code
    assert "d = c * 2" in code
    assert "return d" in code
    assert code.endswith("compute(a, b)")


def test_no_return():
    def noop(x: int) -> None:
        _ = x + 1

    code, input_names = extract_source(noop)
    assert input_names == ["x"]
    assert "def noop" in code
    assert code.endswith("noop(x)")


def test_bare_return():
    def bare(x: int) -> None:
        if x > 0:
            return

    code, input_names = extract_source(bare)
    assert input_names == ["x"]
    assert "def bare" in code
    assert "return" in code
    assert code.endswith("bare(x)")


def test_multiple_returns():
    def branching(x: int) -> int:
        if x > 0:
            return x
        return -x

    code, input_names = extract_source(branching)
    assert input_names == ["x"]
    assert "def branching" in code
    assert "return x" in code
    assert "return -x" in code
    assert code.endswith("branching(x)")


def test_async_function():
    async def fetch(x: int) -> int:
        return x + 1

    code, input_names = extract_source(fetch)
    assert input_names == ["x"]
    assert "async def fetch" in code
    assert "return x + 1" in code
    assert code.endswith("await fetch(x)")


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
    assert "def classify" in code
    assert 'return "positive"' in code
    assert 'return "negative"' in code
    assert 'return "zero"' in code
    assert code.endswith("classify(x)")


def test_string_operations():
    def greet(name: str) -> str:
        return f"hello {name}"

    code, input_names = extract_source(greet)
    assert input_names == ["name"]
    assert "def greet" in code
    assert "return" in code
    assert code.endswith("greet(name)")
