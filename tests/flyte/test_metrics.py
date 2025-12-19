import asyncio

import pytest

from flyte._metrics import async_timer, _timer_stack


@pytest.mark.asyncio
async def test_sibling_timers():
    async def task1():
        async with async_timer("task1"):
            # Stack for this task: ["task1"]
            assert _timer_stack.get() == ["task1"]
            await asyncio.sleep(0.2)
            async with async_timer("inner"):
                # Stack for this task: ["task1", "inner"]
                assert _timer_stack.get() == ["task1", "inner"]
                await asyncio.sleep(0.2)
        assert _timer_stack.get() == []

    async def task2():
        async with async_timer("task2"):
            assert _timer_stack.get() == ["task2"]
            await asyncio.sleep(0.1)
        assert _timer_stack.get() == []

    await asyncio.gather(task1(), task2())
