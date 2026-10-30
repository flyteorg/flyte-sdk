from typing import List

import pytest

import flyte
import flyte.errors


def test_map_partials_happy():
    from functools import partial

    # Define an environment with specific resources
    env = flyte.TaskEnvironment(name="map-test", resources=flyte.Resources(cpu="1"))

    @env.task
    def my_task(batch_id: int, name: str, constant_param: str) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def main() -> List[str]:
        compounds = list(range(100))
        constant_param = "shared_config"

        curry_consts = partial(my_task, constant_param=constant_param, name="daniel")

        return list(flyte.map(curry_consts, compounds))

    flyte.init()
    run = flyte.with_runcontext(mode="local").run(main)
    print(run.outputs())


def test_partial_validation():
    from functools import partial

    from flyte._map import _Mapper

    env = flyte.TaskEnvironment(name="map-test")

    @env.task
    def my_task(name: str, constant_param: str, batch_id: int) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def three_param_task(a: int, b: str, c: float) -> str:
        return f"{a}-{b}-{c}"

    mapper = _Mapper()

    # Case 1: No parameters left for mapping (all parameters provided)
    full_partial = partial(my_task, "alice", "config", 42)
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(full_partial)

    # Case 2: Too many parameters left for mapping (no parameters provided)
    empty_partial = partial(my_task)
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(empty_partial)

    # Case 3: Two parameters left for mapping
    two_left_partial = partial(my_task, "alice")
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(two_left_partial)

    # Case 4: Parameter provided both as positional and keyword argument
    duplicate_param_partial = partial(three_param_task, 1, a=2)
    with pytest.raises(TypeError, match="provided both as positional argument and keyword argument"):
        mapper.validate_partial(duplicate_param_partial)


def test_map_partials_unhappy():
    from functools import partial

    # Define an environment with specific resources
    env = flyte.TaskEnvironment(name="map-test", resources=flyte.Resources(cpu="1"))

    @env.task
    def my_task(name: str, constant_param: str, batch_id: int) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def main() -> List[str]:
        compounds = list(range(100))
        constant_param = "shared_config"

        curry_consts = partial(my_task, constant_param=constant_param, name="daniel")

        return list(flyte.map(curry_consts, compounds))

    flyte.init()
    with pytest.raises(flyte.errors.RuntimeUserError) as excinfo:
        flyte.with_runcontext(mode="local").run(main)
    assert excinfo.value.code == "TypeError", excinfo.value


@pytest.mark.asyncio
async def test_map_async_iterator_initialize_with_partial():
    from functools import partial
    from unittest.mock import AsyncMock, Mock

    from flyte._map import MapAsyncIterator

    # Create a mock TaskTemplate
    mock_task = Mock()
    mock_task.name = "test_task"
    mock_task.aio = AsyncMock()

    # Create a partial with bound arguments (func attribute is automatically set)
    partial_func = partial(mock_task, "bound_arg1", constant="bound_value")

    # Test data for mapping
    map_args = ([1, 2, 3],)  # One iterable to map over

    # Create MapAsyncIterator with partial function
    iterator = MapAsyncIterator(
        func=partial_func, args=map_args, name="test_group", concurrency=0, return_exceptions=True
    )

    # Collect results to trigger task execution
    await iterator.collect()

    # Verify that the correct number of tasks were created and executed
    assert iterator._task_count == 3

    # Verify that aio was called with merged arguments for each mapped value
    expected_calls = [
        (("bound_arg1", 1), {"constant": "bound_value"}),
        (("bound_arg1", 2), {"constant": "bound_value"}),
        (("bound_arg1", 3), {"constant": "bound_value"}),
    ]

    assert mock_task.aio.call_count == 3
    for i, call in enumerate(mock_task.aio.call_args_list):
        args, kwargs = call
        expected_args, expected_kwargs = expected_calls[i]
        assert args == expected_args
        assert kwargs == expected_kwargs


@pytest.mark.asyncio
async def test_map_async_iterator_with_name_fn():
    from unittest.mock import AsyncMock, Mock

    from flyte._map import MapAsyncIterator

    mock_overridden_tasks = []

    def fake_override(*, short_name=None, **kwargs):
        child = Mock()
        child.name = "test_task"
        child.short_name = short_name
        child.aio = AsyncMock(return_value=f"res_{short_name}")
        mock_overridden_tasks.append(child)
        return child

    mock_task = Mock()
    mock_task.name = "test_task"
    mock_task.short_name = "test_task"
    mock_task.override = Mock(side_effect=fake_override)

    map_args = ([1, 2, 3],)
    iterator = MapAsyncIterator(
        func=mock_task,
        args=map_args,
        name="test_group",
        concurrency=0,
        return_exceptions=True,
        name_fn=lambda x: f"item_{x}",
    )

    results = await iterator.collect()
    assert results == ["res_item_1", "res_item_2", "res_item_3"]
    assert mock_task.override.call_count == 3
    assert [c.kwargs["short_name"] for c in mock_task.override.call_args_list] == [
        "item_1",
        "item_2",
        "item_3",
    ]


@pytest.mark.asyncio
async def test_map_async_iterator_name_fn_truncation_and_none():
    from unittest.mock import AsyncMock, Mock

    from flyte._map import MapAsyncIterator

    recorded_names = []

    def fake_override(*, short_name=None, **kwargs):
        recorded_names.append(short_name)
        child = Mock()
        child.aio = AsyncMock(return_value=short_name)
        return child

    mock_task = Mock()
    mock_task.name = "test_task"
    mock_task.short_name = "original_name"
    mock_task.override = Mock(side_effect=fake_override)

    # 1. Name longer than 63 characters should be truncated
    long_name = "a" * 100
    it_long = MapAsyncIterator(
        func=mock_task,
        args=([1],),
        name="test",
        concurrency=0,
        return_exceptions=True,
        name_fn=lambda x: long_name,
    )
    await it_long.collect()
    assert recorded_names[-1] == "a" * 63
    assert len(recorded_names[-1]) == 63

    # 2. When name_fn returns None, original task is kept without override
    mock_task.aio = AsyncMock(return_value="default_res")
    it_none = MapAsyncIterator(
        func=mock_task,
        args=([2],),
        name="test",
        concurrency=0,
        return_exceptions=True,
        name_fn=lambda x: None,
    )
    res = await it_none.collect()
    assert res == ["default_res"]


def test_map_with_name_fn_sync_local():
    env = flyte.TaskEnvironment(name="map-name-fn-test")

    @env.task
    def process_item(item_id: str) -> str:
        return f"processed:{item_id}"

    results = list(
        flyte.map(
            process_item,
            ["x", "y", "z"],
            name_fn=lambda item: f"worker_{item}",
        )
    )
    assert results == ["processed:x", "processed:y", "processed:z"]


@pytest.mark.asyncio
async def test_map_with_name_fn_async_local():
    env = flyte.TaskEnvironment(name="map-name-fn-async-test")

    @env.task
    async def process_async(item_id: str) -> str:
        return f"async:{item_id}"

    results = []
    async for r in flyte.map.aio(
        process_async,
        ["x", "y", "z"],
        name_fn=lambda item: f"worker_{item}",
    ):
        results.append(r)

    assert results == ["async:x", "async:y", "async:z"]


def test_map_with_name_fn_partial_local():
    from functools import partial

    env = flyte.TaskEnvironment(name="map-name-fn-partial-test")

    @env.task
    def process_pair(item_id: str, prefix: str) -> str:
        return f"{prefix}:{item_id}"

    bound = partial(process_pair, prefix="pre")
    results = list(
        flyte.map(
            bound,
            ["1", "2"],
            name_fn=lambda item: f"task_{item}",
        )
    )
    assert results == ["pre:1", "pre:2"]


def test_map_with_name_fn_multi_args():
    env = flyte.TaskEnvironment(name="map-name-fn-multi-test")

    @env.task
    def combine(a: int, b: str) -> str:
        return f"{b}_{a}"

    results = list(
        flyte.map(
            combine,
            [1, 2],
            ["alpha", "beta"],
            name_fn=lambda a, b: f"{b}-{a}",
        )
    )
    assert results == ["alpha_1", "beta_2"]
