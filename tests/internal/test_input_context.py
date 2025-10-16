"""Tests for custom_context functionality."""

import pytest

import flyte
from flyte._context import internal_ctx
from flyte._run import with_runcontext
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report


@pytest.fixture
def task_context():
    """Create a basic task context for testing."""
    return TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"global_key": "global_value"},
    )


def test_get_custom_context_from_task_context(task_context):
    """Test that get_custom_context retrieves context from TaskContext."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        result = flyte.get_custom_context()
        assert result == {"global_key": "global_value"}


def test_get_custom_context_without_task_context():
    """Test that get_custom_context returns empty dict when no context is set."""
    # Ensure no task context is set
    result = flyte.get_custom_context()
    assert result == {}


def test_custom_context_manager_basic(task_context):
    """Test basic custom_context context manager functionality."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        with flyte.custom_context(project="my-project"):
            result = flyte.get_custom_context()
            # Should have both base context and code-provided context
            assert result["global_key"] == "global_value"
            assert result["project"] == "my-project"


def test_custom_context_manager_nested(task_context):
    """Test nested custom_context context managers."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        with flyte.custom_context(project="my-project"):
            with flyte.custom_context(entity="my-entity"):
                result = flyte.get_custom_context()
                # Inner context should have all values
                assert result["global_key"] == "global_value"
                assert result["project"] == "my-project"
                assert result["entity"] == "my-entity"


def test_custom_context_manager_override():
    """Test that code-provided values override base values."""
    tctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"project": "global-project", "env": "production"},
    )

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        with flyte.custom_context(project="my-project"):
            result = flyte.get_custom_context()
            # Code value should override base value
            assert result["project"] == "my-project"
            assert result["env"] == "production"


def test_custom_context(task_context):
    """Test synchronous custom_context context manager."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        with flyte.custom_context(project="my-project"):
            result = flyte.get_custom_context()
            assert result["global_key"] == "global_value"
            assert result["project"] == "my-project"


@pytest.mark.asyncio
async def test_custom_context_isolation(task_context):
    """Test that context is isolated between context managers."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        with flyte.custom_context(project="project1"):
            result1 = flyte.get_custom_context()
            assert result1["project"] == "project1"
            assert result1["global_key"] == "global_value"

        # After exiting, context should be reset to original
        result2 = flyte.get_custom_context()
        assert "project" not in result2
        # Should still have the original task context
        assert result2 == {"global_key": "global_value"}


@pytest.mark.asyncio
async def test_custom_context_with_task_context(task_context):
    """Test that custom_context works correctly with task context."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        # get_custom_context should return task context's custom_context
        result = flyte.get_custom_context()
        assert result == {"global_key": "global_value"}

        # Context manager should still work and merge correctly
        with flyte.custom_context(project="my-project"):
            result_inner = flyte.get_custom_context()
            # Context manager should have merged with base
            assert result_inner.get("project") == "my-project"
            assert result_inner.get("global_key") == "global_value"


@pytest.mark.asyncio
async def test_nested_context_managers_detailed():
    """Test detailed nested context manager behavior with restore."""
    # Initialize with task context
    tctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"env": "production", "region": "us-west-2"},
    )

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        # Level 0: Base context only
        assert flyte.get_custom_context() == {"env": "production", "region": "us-west-2"}

        with flyte.custom_context(project="project1"):
            # Level 1: Base + project1
            ctx1 = flyte.get_custom_context()
            assert ctx1 == {"env": "production", "region": "us-west-2", "project": "project1"}

            with flyte.custom_context(entity="entity1"):
                # Level 2: Base + project1 + entity1
                ctx2 = flyte.get_custom_context()
                assert ctx2 == {
                    "env": "production",
                    "region": "us-west-2",
                    "project": "project1",
                    "entity": "entity1",
                }

                with flyte.custom_context(id="123"):
                    # Level 3: Base + project1 + entity1 + id
                    ctx3 = flyte.get_custom_context()
                    assert ctx3 == {
                        "env": "production",
                        "region": "us-west-2",
                        "project": "project1",
                        "entity": "entity1",
                        "id": "123",
                    }

                # Back to Level 2
                ctx2_after = flyte.get_custom_context()
                assert ctx2_after == {
                    "env": "production",
                    "region": "us-west-2",
                    "project": "project1",
                    "entity": "entity1",
                }
                assert "id" not in ctx2_after

            # Back to Level 1
            ctx1_after = flyte.get_custom_context()
            assert ctx1_after == {"env": "production", "region": "us-west-2", "project": "project1"}
            assert "entity" not in ctx1_after

        # Back to Level 0
        ctx0_after = flyte.get_custom_context()
        assert ctx0_after == {"env": "production", "region": "us-west-2"}
        assert "project" not in ctx0_after


@pytest.mark.asyncio
async def test_nested_context_override():
    """Test that nested contexts can override parent values."""
    tctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"env": "production", "project": "global-project"},
    )

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        with flyte.custom_context(project="parent-project"):
            ctx1 = flyte.get_custom_context()
            assert ctx1["project"] == "parent-project"
            assert ctx1["env"] == "production"

            with flyte.custom_context(project="child-project", entity="my-entity"):
                # Child overrides parent's project
                ctx2 = flyte.get_custom_context()
                assert ctx2["project"] == "child-project"
                assert ctx2["entity"] == "my-entity"
                assert ctx2["env"] == "production"

            # Back to parent context
            ctx1_after = flyte.get_custom_context()
            assert ctx1_after["project"] == "parent-project"
            assert "entity" not in ctx1_after


def test_with_runcontext_basic():
    """Test that with_runcontext sets custom_context correctly in main block usage."""
    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def test_task() -> dict[str, str]:
        ctx = flyte.get_custom_context()
        return ctx

    # Simulate main block: with_runcontext().run() to trigger execution
    result = with_runcontext(mode="local", custom_context={"project": "my-project", "env": "prod"}).run(test_task)
    assert result.outputs() == {"project": "my-project", "env": "prod"}


def test_with_runcontext_child_task_propagation():
    """Test that custom_context propagates from parent to child tasks."""
    from typing import Any

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def child_task() -> dict[str, str]:
        # Child should inherit parent's context
        ctx = flyte.get_custom_context()
        return ctx

    @env.task
    def parent_task() -> dict[str, Any]:
        # Parent should have context from with_runcontext
        parent_ctx = flyte.get_custom_context()
        # Call child task - context should propagate
        child_ctx = child_task()
        return {"parent": parent_ctx, "child": child_ctx}

    # Main block: trigger parent task with context
    result = with_runcontext(mode="local", custom_context={"project": "test-proj", "entity": "test-entity"}).run(
        parent_task
    )
    outputs = result.outputs()
    assert outputs["parent"] == {"project": "test-proj", "entity": "test-entity"}
    assert outputs["child"] == {"project": "test-proj", "entity": "test-entity"}


def test_with_runcontext_multiple_levels():
    """Test context propagation through multiple task levels."""
    from typing import Any

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def grandchild_task() -> dict[str, str]:
        return flyte.get_custom_context()

    @env.task
    def child_task() -> dict[str, Any]:
        child_ctx = flyte.get_custom_context()
        grandchild_ctx = grandchild_task()
        return {"child": child_ctx, "grandchild": grandchild_ctx}

    @env.task
    def parent_task() -> dict[str, Any]:
        parent_ctx = flyte.get_custom_context()
        result = child_task()
        result["parent"] = parent_ctx
        return result

    # Main block: context should propagate through all levels
    result = with_runcontext(mode="local", custom_context={"project": "test", "region": "us-west-2"}).run(parent_task)
    outputs = result.outputs()
    expected = {"project": "test", "region": "us-west-2"}
    assert outputs["parent"] == expected
    assert outputs["child"] == expected
    assert outputs["grandchild"] == expected


def test_with_runcontext_and_context_manager():
    """Test that custom_context context manager works with with_runcontext."""
    from typing import Any, Dict

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    async def task_with_override() -> Dict[str, str]:
        return flyte.get_custom_context()

    @env.task
    async def parent_task() -> Dict[str, Any]:
        # Without override
        ctx1 = await task_with_override()

        # With override via context manager - merges with base context
        with flyte.custom_context(entity="override-entity", new_key="new_value"):
            ctx2 = await task_with_override()

        return {"without_override": ctx1, "with_override": ctx2}

    # Main block: set base context, then parent task uses context manager for overrides
    result = with_runcontext(mode="local", custom_context={"project": "base-project", "entity": "base-entity"}).run(
        parent_task
    )
    outputs = result.outputs()

    # Without override should have base context
    assert outputs["without_override"] == {"project": "base-project", "entity": "base-entity"}

    # With override should merge contexts (context manager values added)
    assert outputs["with_override"] == {
        "project": "base-project",
        "entity": "override-entity",
        "new_key": "new_value",
    }


def test_with_runcontext_empty():
    """Test with_runcontext with no custom_context."""
    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def test_task() -> dict[str, str]:
        return flyte.get_custom_context()

    # Main block: no context provided
    result = with_runcontext(mode="local").run(test_task)
    assert result.outputs() == {}


def test_with_runcontext_parallel_tasks():
    """Test that context propagates correctly to parallel tasks."""
    import asyncio
    from typing import Any

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    async def task1() -> dict[str, Any]:
        return {"task": "task1", "ctx": flyte.get_custom_context()}

    @env.task
    async def task2() -> dict[str, Any]:
        return {"task": "task2", "ctx": flyte.get_custom_context()}

    @env.task
    async def parent_task() -> list[dict[str, Any]]:
        # Execute tasks in parallel
        results = await asyncio.gather(task1(), task2())
        return list(results)

    # Main block: context should propagate to both parallel tasks
    result = with_runcontext(mode="local", custom_context={"project": "parallel-test", "batch": "123"}).run(parent_task)
    outputs = result.outputs()

    # Both tasks should have the same context
    expected = {"project": "parallel-test", "batch": "123"}
    assert outputs[0]["ctx"] == expected
    assert outputs[1]["ctx"] == expected


def test_with_runcontext_isolation():
    """Test that context from one execution doesn't leak to another."""
    from typing import Dict

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def test_task() -> Dict[str, str]:
        return flyte.get_custom_context()

    # First execution with context
    result1 = with_runcontext(mode="local", custom_context={"project": "project1"}).run(test_task)
    assert result1.outputs() == {"project": "project1"}

    # Second execution with different context - should not have first context
    result2 = with_runcontext(mode="local", custom_context={"project": "project2", "env": "staging"}).run(test_task)
    assert result2.outputs() == {"project": "project2", "env": "staging"}

    # Third execution without context
    result3 = with_runcontext(mode="local").run(test_task)
    assert result3.outputs() == {}
