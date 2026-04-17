import asyncio
from datetime import timedelta

import flyte
from flyte.extras import Sleep
import flyte.report

# Leaves run in leaseworker via the core-sleep plugin: no task pods are created,
# so we can fan out wide without paying pod-startup cost.
sleep_env = flyte.TaskEnvironment(
    name="sleep_fanout_leaf",
    plugin_config=Sleep(),
)

fanout_env = flyte.TaskEnvironment(
    name="sleep_fanout",
    resources=flyte.Resources(cpu=1, memory="200Mi"),
    depends_on=[sleep_env],
)

swarm_env = flyte.TaskEnvironment(
    name="sleep_fanout_swarm",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    depends_on=[fanout_env],
)

@sleep_env.task
async def sleep_leaf(duration: timedelta) -> None:
    return None


@fanout_env.task
async def sleep_fanout(
    n_children: int = 10,
    sleep_duration: timedelta = timedelta(seconds=0),
) -> int:
    """
    Fan out n_children core-sleep leaves in parallel.

    All leaves run in leaseworker via the core-sleep plugin, so no task pods
    are created.
    """
    await asyncio.gather(*(sleep_leaf(duration=sleep_duration) for _ in range(n_children)))
    print(f"Done. Total leaves: {n_children}")
    return n_children


@swarm_env.task
async def submit_runs(
    n_runs: int,
    max_rps: int,
    n_children: int,
    sleep_duration: timedelta,
) -> list[str]:
    """
    Submit n_runs sleep_fanout runs rate-limited at max_rps submissions/sec.
    Returns the URLs of every successfully submitted run.
    """
    from aiolimiter import AsyncLimiter

    limiter = AsyncLimiter(max_rps, 1)

    async def submit_one() -> str:
        async with limiter:
            run = await flyte.run.aio(sleep_fanout, n_children=n_children, sleep_duration=sleep_duration)
            return run.url

    urls = await asyncio.gather(*(submit_one() for _ in range(n_runs)))
    print(f"Swarm worker done. Submitted {len(urls)} runs at <= {max_rps} rps.")
    return list(urls)


@swarm_env.task(report=True)
async def main(
    swarm_size: int = 5,
    runs_per_worker: int = 20,
    max_rps: int = 10,
    n_children: int = 10,
    sleep_duration: timedelta = timedelta(seconds=0),
) -> int:
    """
    Spawn `swarm_size` parallel swarm workers; each submits `runs_per_worker`
    sleep_fanout runs, rate-limited at `max_rps` submissions/sec per worker.

    Total submitted runs = swarm_size * runs_per_worker.
    Aggregate submission ceiling ~ swarm_size * max_rps.

    A primer invocation runs first to ensure sleep_env and fanout_env are built
    before the swarm starts hammering them. swarm_env is already warm because
    `main` itself runs there.

    Emits a report with links to every submitted run.
    """
    await sleep_fanout.override(short_name="primer")(n_children=1, sleep_duration=timedelta(seconds=0))

    url_lists = await asyncio.gather(
        *(
            submit_runs(
                n_runs=runs_per_worker,
                max_rps=max_rps,
                n_children=n_children,
                sleep_duration=sleep_duration,
            )
            for _ in range(swarm_size)
        )
    )
    urls = [u for worker_urls in url_lists for u in worker_urls]
    total = swarm_size * runs_per_worker

    tab = flyte.report.get_tab("main")
    tab.log("<h2>Sleep fanout swarm</h2>")
    tab.log(
        f"<ul>"
        f"<li>swarm_size: {swarm_size}</li>"
        f"<li>runs_per_worker: {runs_per_worker}</li>"
        f"<li>max_rps (per worker): {max_rps}</li>"
        f"<li>n_children per run: {n_children}</li>"
        f"<li>sleep_duration: {sleep_duration}</li>"
        f"<li>total submitted: {len(urls)} / {total}</li>"
        f"</ul>"
    )
    tab.log("<h3>Submitted runs</h3><ol>")
    for u in urls:
        tab.log(f'<li><a href="{u}" target="_blank">{u}</a></li>')
    tab.log("</ol>")
    await flyte.report.flush.aio()

    print(f"Done. Swarm={swarm_size} submitted {len(urls)} runs total.")
    return len(urls)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext("remote").run(
        main,
        swarm_size=5,
        runs_per_worker=20,
        max_rps=10,
        n_children=10,
        sleep_duration=timedelta(seconds=30),
    )
    print(run.url)
