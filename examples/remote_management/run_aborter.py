from datetime import datetime, timedelta, timezone

import pytest

import flyte
import flyte.remote as remote

now = datetime.now(timezone.utc)
ten_hours_ago = now - timedelta(hours=10)


async def get_running_runs() -> list[remote.Run]:
    runs = [t async for t in remote.Run.listall.aio(in_phase=("running",))]
    return runs


@pytest.mark.asyncio
async def test_get_run():
    await flyte.init_from_config.aio(
        "/Path/to/your/flyte.config.yaml",
    )
    running = await get_running_runs()

    print(f"Found {len(running)} running runs")

    count = 0
    for r in running:
        if r.action.start_time < ten_hours_ago:
            print(f"Aborting run {r.name} {r.url}", flush=True)
            await r.abort.aio()
            count += 1

    print(f"Found {len(running)} running runs, aborted {count} runs")
