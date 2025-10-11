"""Tests for input_context functionality."""
import pytest

import flyte
from flyte._context import internal_ctx
from flyte._input_context import _input_context_var
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report
from flyte._run import with_runcontext


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
        input_context={"global_key": "global_value"},
    )


def test_get_input_context_from_task_context(task_context):
    """Test that get_input_context retrieves context from TaskContext."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        result = flyte.get_input_context()
        assert result == {"global_key": "global_value"}


def test_get_input_context_without_task_context():
    """Test that get_input_context returns empty dict when no context is set."""
    # Ensure no task context is set
    result = flyte.get_input_context()
    assert result == {}


def test_input_context_manager_basic():
    """Test basic input_context context manager functionality."""
    # Set global context
    token = _input_context_var.set({"env": "production"})

    try:
        async def test_async():
            async with flyte.input_context(project="my-project"):
                context = _input_context_var.get()
                # Should have both global and code-provided context
                assert context["env"] == "production"
                assert context["project"] == "my-project"

        import asyncio
        asyncio.run(test_async())
    finally:
        _input_context_var.reset(token)


def test_input_context_manager_nested():
    """Test nested input_context context managers."""
    token = _input_context_var.set({"env": "production"})

    try:
        async def test_async():
            async with flyte.input_context(project="my-project"):
                async with flyte.input_context(entity="my-entity"):
                    context = _input_context_var.get()
                    # Inner context should have all values
                    assert context["env"] == "production"
                    assert context["project"] == "my-project"
                    assert context["entity"] == "my-entity"

        import asyncio
        asyncio.run(test_async())
    finally:
        _input_context_var.reset(token)


def test_input_context_manager_override():
    """Test that code-provided values override global values."""
    token = _input_context_var.set({"project": "global-project", "env": "production"})

    try:
        async def test_async():
            async with flyte.input_context(project="my-project"):
                context = _input_context_var.get()
                # Code value should override global value
                assert context["project"] == "my-project"
                assert context["env"] == "production"

        import asyncio
        asyncio.run(test_async())
    finally:
        _input_context_var.reset(token)


def test_input_context_sync():
    """Test synchronous input_context_sync context manager."""
    from flyte._input_context import input_context_sync

    token = _input_context_var.set({"env": "production"})

    try:
        with input_context_sync(project="my-project"):
            context = _input_context_var.get()
            assert context["env"] == "production"
            assert context["project"] == "my-project"
    finally:
        _input_context_var.reset(token)


@pytest.mark.asyncio
async def test_input_context_isolation():
    """Test that context is isolated between context managers."""
    token = _input_context_var.set({"env": "production"})

    try:
        async with flyte.input_context(project="project1"):
            context1 = _input_context_var.get()
            assert context1["project"] == "project1"
            assert context1["env"] == "production"

        # After exiting, context should be reset to original
        context2 = _input_context_var.get()
        assert "project" not in context2
        # Should still have the original global context
        assert context2 == {"env": "production"}
    finally:
        _input_context_var.reset(token)


@pytest.mark.asyncio
async def test_input_context_with_task_context(task_context):
    """Test that input_context works correctly with task context."""
    ctx = internal_ctx()
    with ctx.replace_task_context(task_context):
        # get_input_context should return task context's input_context
        result = flyte.get_input_context()
        assert result == {"global_key": "global_value"}

        # Context manager should still work and merge correctly
        async with flyte.input_context(project="my-project"):
            var_context = _input_context_var.get()
            # Context manager should have merged with global
            assert var_context.get("project") == "my-project"


@pytest.mark.asyncio
async def test_nested_context_managers_detailed():
    """Test detailed nested context manager behavior with restore."""
    # Initialize with global context
    token = _input_context_var.set({"env": "production", "region": "us-west-2"})

    try:
        # Level 0: Global context only
        assert _input_context_var.get() == {"env": "production", "region": "us-west-2"}

        async with flyte.input_context(project="project1"):
            # Level 1: Global + project1
            ctx1 = _input_context_var.get()
            assert ctx1 == {"env": "production", "region": "us-west-2", "project": "project1"}

            async with flyte.input_context(entity="entity1"):
                # Level 2: Global + project1 + entity1
                ctx2 = _input_context_var.get()
                assert ctx2 == {
                    "env": "production",
                    "region": "us-west-2",
                    "project": "project1",
                    "entity": "entity1",
                }

                async with flyte.input_context(id="123"):
                    # Level 3: Global + project1 + entity1 + id
                    ctx3 = _input_context_var.get()
                    assert ctx3 == {
                        "env": "production",
                        "region": "us-west-2",
                        "project": "project1",
                        "entity": "entity1",
                        "id": "123",
                    }

                # Back to Level 2
                ctx2_after = _input_context_var.get()
                assert ctx2_after == {
                    "env": "production",
                    "region": "us-west-2",
                    "project": "project1",
                    "entity": "entity1",
                }
                assert "id" not in ctx2_after

            # Back to Level 1
            ctx1_after = _input_context_var.get()
            assert ctx1_after == {"env": "production", "region": "us-west-2", "project": "project1"}
            assert "entity" not in ctx1_after

        # Back to Level 0
        ctx0_after = _input_context_var.get()
        assert ctx0_after == {"env": "production", "region": "us-west-2"}
        assert "project" not in ctx0_after

    finally:
        _input_context_var.reset(token)


@pytest.mark.asyncio
async def test_nested_context_override():
    """Test that nested contexts can override parent values."""
    token = _input_context_var.set({"env": "production", "project": "global-project"})

    try:
        async with flyte.input_context(project="parent-project"):
            ctx1 = _input_context_var.get()
            assert ctx1["project"] == "parent-project"
            assert ctx1["env"] == "production"

            async with flyte.input_context(project="child-project", entity="my-entity"):
                # Child overrides parent's project
                ctx2 = _input_context_var.get()
                assert ctx2["project"] == "child-project"
                assert ctx2["entity"] == "my-entity"
                assert ctx2["env"] == "production"

            # Back to parent context
            ctx1_after = _input_context_var.get()
            assert ctx1_after["project"] == "parent-project"
            assert "entity" not in ctx1_after

    finally:
        _input_context_var.reset(token)


def test_with_runcontext_basic():
    """Test that with_runcontext sets input_context correctly in main block usage."""
    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def test_task() -> dict[str, str]:
        ctx = flyte.get_input_context()
        return ctx

    # Simulate main block: with_runcontext().run() to trigger execution
    result = with_runcontext(mode="local", input_context={"project": "my-project", "env": "prod"}).run(test_task)
    assert result.outputs() == {"project": "my-project", "env": "prod"}


def test_with_runcontext_child_task_propagation():
    """Test that input_context propagates from parent to child tasks."""
    from typing import Any

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def child_task() -> dict[str, str]:
        # Child should inherit parent's context
        ctx = flyte.get_input_context()
        return ctx

    @env.task
    def parent_task() -> dict[str, Any]:
        # Parent should have context from with_runcontext
        parent_ctx = flyte.get_input_context()
        # Call child task - context should propagate
        child_ctx = child_task()
        return {"parent": parent_ctx, "child": child_ctx}

    # Main block: trigger parent task with context
    result = with_runcontext(mode="local", input_context={"project": "test-proj", "entity": "test-entity"}).run(
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
        return flyte.get_input_context()

    @env.task
    def child_task() -> dict[str, Any]:
        child_ctx = flyte.get_input_context()
        grandchild_ctx = grandchild_task()
        return {"child": child_ctx, "grandchild": grandchild_ctx}

    @env.task
    def parent_task() -> dict[str, Any]:
        parent_ctx = flyte.get_input_context()
        result = child_task()
        result["parent"] = parent_ctx
        return result

    # Main block: context should propagate through all levels
    result = with_runcontext(mode="local", input_context={"project": "test", "region": "us-west-2"}).run(parent_task)
    outputs = result.outputs()
    expected = {"project": "test", "region": "us-west-2"}
    assert outputs["parent"] == expected
    assert outputs["child"] == expected
    assert outputs["grandchild"] == expected


def test_with_runcontext_and_context_manager():
    """Test that input_context context manager works with with_runcontext."""
    from typing import Any, Dict

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    async def task_with_override() -> Dict[str, str]:
        return flyte.get_input_context()

    @env.task
    async def parent_task() -> Dict[str, Any]:
        # Without override
        ctx1 = await task_with_override()

        # With override via context manager - merges with base context
        async with flyte.input_context(entity="override-entity", new_key="new_value"):
            ctx2 = await task_with_override()

        return {"without_override": ctx1, "with_override": ctx2}

    # Main block: set base context, then parent task uses context manager for overrides
    result = with_runcontext(mode="local", input_context={"project": "base-project", "entity": "base-entity"}).run(parent_task)
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
    """Test with_runcontext with no input_context."""
    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    def test_task() -> dict[str, str]:
        return flyte.get_input_context()

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
        return {"task": "task1", "ctx": flyte.get_input_context()}

    @env.task
    async def task2() -> dict[str, Any]:
        return {"task": "task2", "ctx": flyte.get_input_context()}

    @env.task
    async def parent_task() -> list[dict[str, Any]]:
        # Execute tasks in parallel
        results = await asyncio.gather(task1(), task2())
        return list(results)

    # Main block: context should propagate to both parallel tasks
    result = with_runcontext(mode="local", input_context={"project": "parallel-test", "batch": "123"}).run(
        parent_task
    )
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
        return flyte.get_input_context()

    # First execution with context
    result1 = with_runcontext(mode="local", input_context={"project": "project1"}).run(test_task)
    assert result1.outputs() == {"project": "project1"}

    # Second execution with different context - should not have first context
    result2 = with_runcontext(mode="local", input_context={"project": "project2", "env": "staging"}).run(test_task)
    assert result2.outputs() == {"project": "project2", "env": "staging"}

    # Third execution without context
    result3 = with_runcontext(mode="local").run(test_task)
    assert result3.outputs() == {}
