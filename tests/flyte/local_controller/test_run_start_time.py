import asyncio
import time
from pathlib import Path

import pytest

import flyte

env = flyte.TaskEnvironment(name="local-run-clock")


def current_run_start() -> str:
    context = flyte.ctx()
    assert context and context.run_start_time is not None
    return context.run_start_time.isoformat()


@env.task
def sync_child() -> str:
    return current_run_start()


@env.task
def sync_parent() -> tuple[str, str]:
    started = current_run_start()
    time.sleep(0.01)
    return started, sync_child()


@env.task
async def async_child() -> str:
    return current_run_start()


@env.task
async def async_parent() -> tuple[str, str]:
    started = current_run_start()
    await asyncio.sleep(0.01)
    return started, await async_child()


@pytest.mark.parametrize("async_task", [False, True], ids=["sync-task", "async-task"])
def test_local_tasks_share_run_start_time(tmp_path: Path, async_task: bool) -> None:
    flyte.init(local_tracked=False)
    runner = flyte.with_runcontext(
        mode="local",
        disable_run_cache=True,
        raw_data_path=str(tmp_path / "raw"),
        run_base_dir=str(tmp_path / "metadata"),
    )
    parent_task = async_parent if async_task else sync_parent
    first = runner.run(parent_task).outputs()
    second = runner.run(parent_task).outputs()

    assert first[0] == first[1]
    assert second[0] == second[1]
    assert first[0] != second[0]


@pytest.mark.asyncio
async def test_async_local_run_preserves_run_start_time(tmp_path: Path) -> None:
    await flyte.init.aio(local_tracked=False)
    run = await flyte.with_runcontext(
        mode="local",
        disable_run_cache=True,
        raw_data_path=str(tmp_path / "raw"),
        run_base_dir=str(tmp_path / "metadata"),
    ).run.aio(async_parent)
    outputs = await run.outputs.aio()

    assert outputs[0] == outputs[1]


def test_local_retries_preserve_run_start_time(tmp_path: Path) -> None:
    starts: list[str] = []

    @env.task(retries=1)
    def retry_once() -> str:
        starts.append(current_run_start())
        if len(starts) == 1:
            raise RuntimeError("retry the synthetic task")
        return starts[-1]

    flyte.init(local_tracked=False)
    run = flyte.with_runcontext(
        mode="local",
        disable_run_cache=True,
        raw_data_path=str(tmp_path / "raw"),
        run_base_dir=str(tmp_path / "metadata"),
    ).run(retry_once)

    assert len(starts) == 2
    assert starts[0] == starts[1] == run.outputs()[0]
