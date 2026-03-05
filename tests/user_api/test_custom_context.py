import pytest

import flyte

env = flyte.TaskEnvironment(name="custom_ctx_test")


@env.task
async def task_reads_context() -> dict:
    return flyte.get_custom_context()


@env.task
async def task_sets_context() -> dict:
    with flyte.custom_context(project="my-project", entity="my-entity"):
        return flyte.get_custom_context()


def test_get_custom_context_outside_task():
    result = flyte.get_custom_context()
    assert result == {}


@pytest.mark.asyncio
async def test_get_custom_context_inside_task():
    await flyte.init.aio()
    run = flyte.run(task_reads_context)
    run.wait()
    assert run.outputs()[0] == {}


@pytest.mark.asyncio
async def test_custom_context_sets_values():
    await flyte.init.aio()
    run = flyte.run(task_sets_context)
    run.wait()
    result = run.outputs()[0]
    assert result["project"] == "my-project"
    assert result["entity"] == "my-entity"
