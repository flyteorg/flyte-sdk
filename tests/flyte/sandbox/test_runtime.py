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


def _double(x: int) -> int:
    return x * 2


def _run(code: str, with_tool: bool):
    # Attaching a tool routes the code through the external-function bridge,
    # which resumes OS calls itself rather than leaving them to Monty.
    tasks = [_double] if with_tool else None
    return asyncio.run(flyte.sandbox.orchestrate_local(code, inputs={}, tasks=tasks))


@pytest.mark.parametrize("with_tool", [False, True], ids=["plain", "with_tool"])
class TestNoClockOrEntropy:
    @pytest.mark.parametrize(
        "code",
        [
            "import time\ntime.time()",
            "import time\ntime.monotonic()",
            "import time\ntime.perf_counter()",
            "import datetime\ndatetime.datetime.now()",
            "import datetime\ndatetime.date.today()",
        ],
    )
    def test_clock_is_refused(self, code, with_tool):
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="not supported in this environment"):
            _run(code, with_tool)

    @pytest.mark.parametrize(
        "code",
        [
            "import random\nrandom.random()",
            "import random\nrandom.randint(1, 10)",
            "import random\nrandom.choice([1, 2, 3])",
            "import random\nrandom.Random().random()",
            "import os\nos.urandom(4)",
        ],
    )
    def test_entropy_is_refused(self, code, with_tool):
        with pytest.raises(pydantic_monty.MontyRuntimeError, match="not supported in this environment"):
            _run(code, with_tool)

    def test_refusal_can_be_caught_in_the_sandbox(self, with_tool):
        code = "import time\ntry:\n    time.time()\n    r = 'has clock'\nexcept RuntimeError:\n    r = 'no clock'\nr"
        assert _run(code, with_tool) == "no clock"

    def test_seeded_random_is_reproducible(self, with_tool):
        code = "import random\nrandom.seed(7)\nrandom.random()"
        assert _run(code, with_tool) == _run(code, with_tool)

    def test_date_arithmetic_still_works(self, with_tool):
        code = "import datetime\nstr(datetime.date(2026, 1, 31) + datetime.timedelta(days=1))"
        assert _run(code, with_tool) == "2026-02-01"

    def test_sleep_still_works(self, with_tool):
        assert _run("import time\ntime.sleep(0.05)\n'slept'", with_tool) == "slept"
