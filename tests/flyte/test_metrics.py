import asyncio
import time

import pytest

from flyte._metrics import Stopwatch


def test_stopwatch_basic():
    """Test basic stopwatch functionality."""
    sw = Stopwatch("test_metric")
    sw.start()
    time.sleep(0.1)
    sw.stop()


def test_stopwatch_not_started():
    """Test that stopping a stopwatch that was never started raises an error."""
    sw = Stopwatch("test_metric")
    with pytest.raises(RuntimeError, match="was never started"):
        sw.stop()


def test_stopwatch_with_extra_fields():
    """Test stopwatch with extra fields."""
    sw = Stopwatch("test_metric", extra_fields={"foo": "bar"})
    sw.start()
    time.sleep(0.1)
    sw.stop()


@pytest.mark.asyncio
async def test_stopwatch_with_async_code():
    """Test that stopwatch works with async code."""
    sw = Stopwatch("async_metric")
    sw.start()
    await asyncio.sleep(0.1)
    sw.stop()
