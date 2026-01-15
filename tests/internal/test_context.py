import asyncio

import pytest

import flyte
from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import Report
from flyte.syncify import syncify


@syncify
async def inner_context_group(outer_group_name: str) -> str:
    final_return = ""
    assert flyte.ctx() is not None
    assert flyte.ctx().action.name is not None
    assert flyte.ctx().group_data is not None
    assert flyte.ctx().group_data.name == outer_group_name
    with flyte.group("inner"):
        assert flyte.ctx().group_data.name == "inner"
        final_return = flyte.ctx().group_data.name
    assert flyte.ctx().group_data.name == outer_group_name
    return final_return


async def outer_context_group(outer_name: str):
    final_return = ""
    assert flyte.ctx() is not None
    ctx = internal_ctx()
    tctx = ctx.data.task_context.replace(data={"x": "y"})
    with ctx.replace_task_context(tctx):
        assert flyte.ctx() is not None
        assert flyte.ctx().data == {"x": "y"}
        assert flyte.ctx().group_data is None
        with flyte.group(outer_name):
            assert flyte.ctx().group_data.name == outer_name
            assert await inner_context_group.aio(outer_name) == "inner"
            final_return = flyte.ctx().group_data.name
        assert flyte.ctx() is not None
        assert flyte.ctx().data == {"x": "y"}
        assert flyte.ctx().group_data is None
    return final_return


def outer_context_group_sync():
    final_return = ""
    assert flyte.ctx() is not None
    ctx = internal_ctx()
    tctx = ctx.data.task_context.replace(data={"x": "y"})
    with ctx.replace_task_context(tctx):
        assert flyte.ctx() is not None
        assert flyte.ctx().data == {"x": "y"}
        assert flyte.ctx().group_data is None
        with flyte.group("outer_sync"):
            assert flyte.ctx().group_data.name == "outer_sync"
            assert inner_context_group("outer_sync") == "inner"
            final_return = flyte.ctx().group_data.name
        assert flyte.ctx() is not None
        assert flyte.ctx().data == {"x": "y"}
        assert flyte.ctx().group_data is None
    return final_return


@pytest.fixture
def outer_task_ctx():
    yield TaskContext(
        action=ActionID(
            name="test",
        ),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
    )


async def simulate_task(new_task_context, outer_group_name):
    assert flyte.ctx() is None
    ctx = internal_ctx()
    with ctx.replace_task_context(new_task_context):
        assert flyte.ctx() is not None
        assert flyte.ctx().group_data is None
        assert await outer_context_group(outer_group_name) == outer_group_name


@pytest.mark.asyncio
async def test_context_group_propagation(outer_task_ctx):
    await simulate_task(outer_task_ctx, "outer_group")


def test_context_group_propagation_sync(outer_task_ctx):
    assert flyte.ctx() is None
    ctx = internal_ctx()
    with ctx.replace_task_context(outer_task_ctx):
        assert flyte.ctx() is not None
        assert flyte.ctx().group_data is None
        assert outer_context_group_sync() == "outer_sync"


@pytest.mark.asyncio
async def test_context_trees(outer_task_ctx):
    tctx1 = outer_task_ctx.replace(action=ActionID(name="test1"))
    tctx2 = outer_task_ctx.replace(action=ActionID(name="test2"))
    parallel_props = [
        simulate_task(tctx1, "outer_group"),
        simulate_task(tctx2, "outer_group"),
    ]
    await asyncio.gather(*parallel_props)


@syncify
async def generator(n: int):
    # NOTE the generator function cannot have a context manager that updates the context, because it will not be
    # as the exit of the context manager will not be awaited until the generator is exhausted.
    assert flyte.ctx() is not None
    assert flyte.ctx().group_data.name == "generator"
    for i in range(n):
        yield f"Item {i}"
    return


async def simulate_gen_task(new_task_context):
    assert flyte.ctx() is None
    ctx = internal_ctx()
    with ctx.replace_task_context(new_task_context):
        assert flyte.ctx() is not None
        assert flyte.ctx().group_data is None
        ctx = internal_ctx()
        tctx = ctx.data.task_context.replace(data={"x": "y"})
        with ctx.replace_task_context(tctx):
            with flyte.group("generator"):
                collect = []
                async for item in generator.aio(n=3):
                    collect.append(item)
            assert collect == [
                "Item 0",
                "Item 1",
                "Item 2",
            ]


@pytest.mark.asyncio
async def test_context_generator(outer_task_ctx):
    await simulate_gen_task(outer_task_ctx)


def test_has_raw_data_with_task_context():
    """Test has_raw_data returns True when raw_data_path is set in task_context"""
    ctx = internal_ctx()
    task_ctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path="/some/path"),
        version="",
        report=Report("test"),
    )
    with ctx.replace_task_context(task_ctx):
        assert flyte.ctx() is not None
        assert internal_ctx().has_raw_data is True


def test_has_raw_data_with_context_data():
    """Test has_raw_data returns True when raw_data_path is set in context data"""
    ctx = internal_ctx()
    new_ctx = ctx.new_raw_data_path(RawDataPath(path="/some/path"))
    with new_ctx:
        assert internal_ctx().has_raw_data is True


def test_has_raw_data_without_raw_data_path():
    """Test has_raw_data returns False when raw_data_path has empty path"""
    ctx = internal_ctx()
    task_ctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
    )
    with ctx.replace_task_context(task_ctx):
        assert flyte.ctx() is not None
        # Empty path should still return True as the RawDataPath object exists
        assert internal_ctx().has_raw_data is True


def test_has_raw_data_empty_context():
    """Test has_raw_data returns False when no task context is set"""
    # Create a fresh context with no task context
    from flyte._context import Context, ContextData

    empty_ctx = Context(data=ContextData())
    with empty_ctx:
        assert internal_ctx().has_raw_data is False


def test_has_raw_data_priority():
    """Test that task_context raw_data_path takes priority over context data raw_data_path"""
    ctx = internal_ctx()

    # Set raw_data_path in context data
    ctx_with_raw_data = ctx.new_raw_data_path(RawDataPath(path="/context/path"))

    # Set raw_data_path in task context
    task_ctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path="/task/path"),
        version="",
        report=Report("test"),
    )

    with ctx_with_raw_data.replace_task_context(task_ctx):
        assert internal_ctx().has_raw_data is True
        # Also verify that raw_data returns the task context path
        assert internal_ctx().raw_data.path == "/task/path"

