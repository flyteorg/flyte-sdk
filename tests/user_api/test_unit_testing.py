import pytest

import flyte

env = flyte.TaskEnvironment("unit_test_env")


@env.task
def add(a: int, b: int) -> int:
    return a + b


@env.task
async def subtract(a: int, b: int) -> int:
    return a - b


@flyte.trace
async def traced_multiply(a: int, b: int) -> int:
    return a * b


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
