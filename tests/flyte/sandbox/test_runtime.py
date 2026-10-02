"""Tests for the SandboxedConfig -> Monty resource limit translation."""

import asyncio
import time

import pytest

import flyte.sandbox
from flyte.sandbox._config import SandboxedConfig
from flyte.sandbox._runtime import _limits

pydantic_monty = pytest.importorskip("pydantic_monty")


class TestLimits:
    def test_keys_are_accepted_by_monty(self):
        # Monty rejects unknown limit keys at checkout, so a renamed key (as
        # `max_duration_secs` was in 1.0) fails here rather than at first use.
        limits = _limits(SandboxedConfig())
        assert set(limits) <= set(pydantic_monty.ResourceLimits.__annotations__)

    def test_timeout_bounds_both_execution_and_sleep(self):
        limits = _limits(SandboxedConfig(timeout_ms=2_500))
        assert limits["max_feed_duration_secs"] == 2.5
        assert limits["max_total_sleep_secs"] == 2.5

    def test_memory_and_recursion_are_forwarded(self):
        limits = _limits(SandboxedConfig(max_memory=1024, max_stack_depth=64))
        assert limits["max_memory"] == 1024
        assert limits["max_recursion_depth"] == 64


class TestTimeoutEnforcement:
    def test_busy_loop_is_stopped(self):
        code = "i = 0\nwhile True:\n    i = i + 1\ni"
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="TimeoutError"):
            asyncio.run(flyte.sandbox.orchestrate_local(code, inputs={}, timeout_ms=200))

    def test_sleep_cannot_outlast_the_timeout(self):
        # Sleeps are excluded from Monty's execution clock; without a sleep
        # budget this snippet would idle for 5s under a 200ms timeout.
        code = "import time\ntime.sleep(5)\n'slept'"
        start = time.monotonic()
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="TimeoutError"):
            asyncio.run(flyte.sandbox.orchestrate_local(code, inputs={}, timeout_ms=200))
        assert time.monotonic() - start < 3

    def test_short_sleep_within_budget_is_allowed(self):
        code = "import time\ntime.sleep(0.05)\n'slept'"
        assert asyncio.run(flyte.sandbox.orchestrate_local(code, inputs={}, timeout_ms=2_000)) == "slept"

    def test_pool_is_usable_after_a_timeout(self):
        async def run() -> int:
            with pytest.raises(pydantic_monty.MontyRuntimeError):
                await flyte.sandbox.orchestrate_local("while True:\n    pass", inputs={}, timeout_ms=200)
            return await flyte.sandbox.orchestrate_local("x + 1", inputs={"x": 1})

        assert asyncio.run(run()) == 2
