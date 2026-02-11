import time
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from flyte.durable import now as durable_now
from flyte.durable import sleep as durable_sleep
from flyte.durable import time as durable_time

# -- durable_time tests --


@pytest.mark.asyncio
async def test_durable_time_returns_float():
    result = await durable_time.aio()
    assert isinstance(result, float)


def test_durable_time_sync_returns_float():
    result = durable_time()
    assert isinstance(result, float)


@pytest.mark.asyncio
async def test_durable_time_returns_current_time():
    before = time.time()
    result = await durable_time.aio()
    after = time.time()
    assert before <= result <= after


def test_durable_time_sync_returns_current_time():
    before = time.time()
    result = durable_time()
    after = time.time()
    assert before <= result <= after


# -- durable_now tests --


@pytest.mark.asyncio
async def test_durable_now_returns_datetime():
    result = await durable_now.aio()
    assert isinstance(result, datetime)


def test_durable_now_sync_returns_datetime():
    result = durable_now()
    assert isinstance(result, datetime)


@pytest.mark.asyncio
async def test_durable_now_returns_current_time():
    before = datetime.now()
    result = await durable_now.aio()
    after = datetime.now()
    assert before <= result <= after


def test_durable_now_sync_returns_current_time():
    before = datetime.now()
    result = durable_now()
    after = datetime.now()
    assert before <= result <= after


# -- durable_sleep tests --


@pytest.mark.asyncio
async def test_durable_sleep_sleeps_for_full_duration():
    """When no wall-clock time has elapsed between sleep_start and now, sleep the full duration."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_durable_sleep_skips_when_time_already_elapsed():
    """When the sleep window has already passed, return immediately without sleeping."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 200.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_sleep_sleeps_only_remaining_time():
    """When some wall-clock time has elapsed, sleep only the remaining amount."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 103.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_durable_sleep_zero_seconds():
    """Sleeping for zero seconds should not invoke asyncio.sleep."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(0)
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_sleep_exact_boundary():
    """When sleep_until == now, no sleep should occur (not strictly greater)."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 105.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_not_awaited()


def test_durable_sleep_sync_sleeps_for_full_duration():
    """Sync interface: sleep the full duration when no time has elapsed."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        durable_sleep(5.0)
        mock_sleep.assert_awaited_once_with(5.0)


def test_durable_sleep_sync_skips_when_time_already_elapsed():
    """Sync interface: return immediately when sleep window has passed."""
    with (
        patch("flyte.durable._time.time.time", side_effect=[100.0, 200.0]),
        patch("flyte.durable._time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        durable_sleep(5.0)
        mock_sleep.assert_not_awaited()
