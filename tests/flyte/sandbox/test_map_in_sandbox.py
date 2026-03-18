"""Tests for mapping over tasks inside sandboxed code strings.

Monty sandbox cannot ``import flyte``. Python's built-in ``map(external_fn, iterable)``
is also unsupported when the function is an external Monty reference.

Two patterns are tested:

1. **For-loop pattern** — ``for item in items: result = task(item)``
   Works today via ``FunctionSnapshot`` pause/resume. Sequential but functional.

2. **flyte_map built-in** — ``flyte_map("task_name", iterable)``
   A bridge-level built-in that resolves the task name and runs ``flyte.map``.
   The bridge handles this as a single ``FunctionSnapshot`` call and returns
   all results at once.
"""

import pytest

import flyte

try:
    from pydantic_monty import Monty  # noqa: F401

    HAS_MONTY = True
except ImportError:
    HAS_MONTY = False


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env():
    return flyte.TaskEnvironment(name="map-sandbox-test")


# ---------------------------------------------------------------------------
# Pattern 1: For-loop calling tasks individually (works today)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestForLoopPatternInSandbox:
    """For-loop pattern: call the task per-item. Works via FunctionSnapshot."""

    @pytest.mark.asyncio
    async def test_for_loop_basic(self, env):
        """Calling a task in a for-loop should work via pause/resume."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            results = []
            for item in items:
                results.append(double(item))
            results
            """,
            inputs={"items": [1, 2, 3]},
            tasks=[double],
        )
        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_for_loop_multiple_args(self, env):
        """For-loop with zip and multi-arg task calls."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def add(x: int, y: int) -> int:
            return x + y

        result = await orchestrate_local(
            """
            results = []
            for x, y in zip(xs, ys):
                results.append(add(x, y))
            results
            """,
            inputs={"xs": [1, 2, 3], "ys": [10, 20, 30]},
            tasks=[add],
        )
        assert result == [11, 22, 33]

    @pytest.mark.asyncio
    async def test_for_loop_empty_iterable(self, env):
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            results = []
            for item in items:
                results.append(double(item))
            results
            """,
            inputs={"items": []},
            tasks=[double],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_for_loop_with_aggregation(self, env):
        """Results from loop can be aggregated."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def square(x: int) -> int:
            return x * x

        result = await orchestrate_local(
            """
            total = 0
            for item in items:
                total = total + square(item)
            total
            """,
            inputs={"items": [1, 2, 3, 4]},
            tasks=[square],
        )
        assert result == 30  # 1 + 4 + 9 + 16

    @pytest.mark.asyncio
    async def test_for_loop_mixed_with_direct_calls(self, env):
        """For-loop + direct task calls in same code string."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        @env.sandbox.orchestrator
        def add(x: int, y: int) -> int:
            return x + y

        result = await orchestrate_local(
            """
            doubled = []
            for item in items:
                doubled.append(double(item))
            add(doubled[0], doubled[1])
            """,
            inputs={"items": [3, 7]},
            tasks=[double, add],
        )
        assert result == 20  # double(3) + double(7) = 6 + 14


# ---------------------------------------------------------------------------
# Pattern 2: flyte_map built-in (bridge resolves task name + runs flyte.map)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestFlyteMapBuiltinInSandbox:
    """flyte_map("task_name", iterable) as a bridge-level built-in.

    The bridge intercepts the ``flyte_map`` call, resolves the task name
    from the registered external refs, and delegates to ``flyte.map``.
    """

    @pytest.mark.asyncio
    async def test_flyte_map_basic(self, env):
        """flyte_map should map a task over items and return results."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            list(flyte_map("double", items))
            """,
            inputs={"items": [1, 2, 3]},
            tasks=[double],
        )
        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_flyte_map_with_aggregation(self, env):
        """flyte_map results can be further processed."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def square(x: int) -> int:
            return x * x

        result = await orchestrate_local(
            """
            total = 0
            for r in flyte_map("square", items):
                total = total + r
            total
            """,
            inputs={"items": [1, 2, 3, 4]},
            tasks=[square],
        )
        assert result == 30

    @pytest.mark.asyncio
    async def test_flyte_map_empty(self, env):
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            list(flyte_map("double", items))
            """,
            inputs={"items": []},
            tasks=[double],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_flyte_map_unknown_task_raises(self, env):
        """Referencing an unknown task name should raise a clear error."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        with pytest.raises(RuntimeError, match=r"unknown.*task|not found"):
            await orchestrate_local(
                """
                list(flyte_map("nonexistent", items))
                """,
                inputs={"items": [1, 2, 3]},
                tasks=[double],
            )

    @pytest.mark.asyncio
    async def test_flyte_map_with_other_calls(self, env):
        """flyte_map coexists with direct task calls."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        @env.sandbox.orchestrator
        def add(x: int, y: int) -> int:
            return x + y

        result = await orchestrate_local(
            """
            doubled = list(flyte_map("double", items))
            add(doubled[0], doubled[1])
            """,
            inputs={"items": [3, 7]},
            tasks=[double, add],
        )
        assert result == 20

    @pytest.mark.asyncio
    async def test_flyte_map_multiple_iterables(self, env):
        """flyte_map with multiple iterables zips them like flyte.map."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def add(x: int, y: int) -> int:
            return x + y

        result = await orchestrate_local(
            """
            flyte_map("add", xs, ys)
            """,
            inputs={"xs": [1, 2, 3], "ys": [10, 20, 30]},
            tasks=[add],
        )
        assert result == [11, 22, 33]

    @pytest.mark.asyncio
    async def test_flyte_map_return_exceptions(self, env):
        """return_exceptions kwarg is forwarded to flyte.map."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def maybe_fail(x: int) -> int:
            if x == 2:
                raise ValueError("bad value")
            return x * 10

        result = await orchestrate_local(
            """
            flyte_map("maybe_fail", items, return_exceptions=True)
            """,
            inputs={"items": [1, 2, 3]},
            tasks=[maybe_fail],
        )
        assert result[0] == 10
        assert isinstance(result[1], Exception)
        assert result[2] == 30

    @pytest.mark.asyncio
    async def test_flyte_map_concurrency_kwarg(self, env):
        """concurrency kwarg is accepted and forwarded."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            flyte_map("double", items, concurrency=2)
            """,
            inputs={"items": [1, 2, 3, 4]},
            tasks=[double],
        )
        assert result == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_flyte_map_group_name_kwarg(self, env):
        """group_name kwarg is accepted and forwarded."""
        from flyte.sandbox import orchestrate_local

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        result = await orchestrate_local(
            """
            flyte_map("double", items, group_name="my-batch")
            """,
            inputs={"items": [1, 2, 3]},
            tasks=[double],
        )
        assert result == [2, 4, 6]


# ---------------------------------------------------------------------------
# orchestrator_from_str — reusable templates with map
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MONTY, reason="pydantic-monty not installed")
class TestMapInOrchestratorFromStr:
    """Templates created via orchestrator_from_str with map patterns."""

    def test_for_loop_template_creates(self, env):
        from flyte.sandbox import orchestrator_from_str

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        t = orchestrator_from_str(
            """
            results = []
            for item in items:
                results.append(double(item))
            results
            """,
            inputs={"items": list},
            output=list,
            tasks=[double],
            name="for-loop-map",
        )
        assert t.name == "for-loop-map"
        assert t._has_external_refs

    def test_flyte_map_template_creates(self, env):
        from flyte.sandbox import orchestrator_from_str

        @env.sandbox.orchestrator
        def double(x: int) -> int:
            return x * 2

        t = orchestrator_from_str(
            """
            list(flyte_map("double", items))
            """,
            inputs={"items": list},
            output=list,
            tasks=[double],
            name="flyte-map-example",
        )
        assert t.name == "flyte-map-example"
        assert t._has_external_refs
