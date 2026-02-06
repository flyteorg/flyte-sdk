from typing import Tuple

import pytest

import flyte
import flyte.errors

env = flyte.TaskEnvironment("unit_test_env")


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
def nested(a: int, b: int) -> int:
    return add(a, b)


@env.task
async def subtract(a: int, b: int) -> int:
    return a - b


@flyte.trace
async def traced_multiply(a: int, b: int) -> int:
    return a * b


@env.task
def tuple_types(x: Tuple[str, str]) -> str:
    return x[0]


def test_add():
    result = add(a=3, b=5)
    assert result == 8


@pytest.mark.asyncio
async def test_subtract():
    result = await subtract(a=10, b=4)
    assert result == 6


@pytest.mark.asyncio
async def test_traced_multiply():
    result = await traced_multiply(a=6, b=7)
    assert result == 42


@pytest.mark.asyncio
async def test_tuple_types():
    tuple_types(x=("a", "b"))
    flyte.run(tuple_types, x=("a", "b"))


@pytest.mark.asyncio
async def test_add_run():
    v = flyte.run(add, 3, 5)
    assert v.outputs()[0] == 8


@pytest.mark.asyncio
async def test_nested_run():
    v = flyte.run(nested, 3, 5)
    assert v.outputs()[0] == 8


@pytest.mark.asyncio
async def test_nested_native():
    v = nested(3, 5)
    assert v == 8
