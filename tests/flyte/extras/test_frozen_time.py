import time
from unittest.mock import AsyncMock, patch

import pytest

from flyte.extras import durable_sleep, durable_time


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


# -- durable_sleep tests --


@pytest.mark.asyncio
async def test_durable_sleep_sleeps_for_full_duration():
    """When no wall-clock time has elapsed between sleep_start and now, sleep the full duration."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_durable_sleep_skips_when_time_already_elapsed():
    """When the sleep window has already passed, return immediately without sleeping."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 200.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_sleep_sleeps_only_remaining_time():
    """When some wall-clock time has elapsed, sleep only the remaining amount."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 103.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_durable_sleep_zero_seconds():
    """Sleeping for zero seconds should not invoke asyncio.sleep."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(0)
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_sleep_exact_boundary():
    """When sleep_until == now, no sleep should occur (not strictly greater)."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 105.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await durable_sleep.aio(5.0)
        mock_sleep.assert_not_awaited()


def test_durable_sleep_sync_sleeps_for_full_duration():
    """Sync interface: sleep the full duration when no time has elapsed."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 100.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        durable_sleep(5.0)
        mock_sleep.assert_awaited_once_with(5.0)


def test_durable_sleep_sync_skips_when_time_already_elapsed():
    """Sync interface: return immediately when sleep window has passed."""
    with (
        patch("flyte.extras._frozen_time.time.time", side_effect=[100.0, 200.0]),
        patch("flyte.extras._frozen_time.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        durable_sleep(5.0)
        mock_sleep.assert_not_awaited()