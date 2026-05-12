import asyncio
import os
from datetime import timedelta

import flyte
import flyte.report
from flyte.extras import Sleep

_STRESS_IMAGE_REGISTRY = os.getenv("FLYTE_STRESS_IMAGE_REGISTRY")
_STRESS_IMAGE_NAME = os.getenv("FLYTE_STRESS_IMAGE_NAME")
_STRESS_IMAGE_PLATFORMS = tuple(
    p.strip() for p in os.getenv("FLYTE_STRESS_IMAGE_PLATFORMS", "linux/amd64").split(",") if p.strip()
)
_STRESS_RUNTIME_ENV = {
    k: v
    for k, v in {
        "FLYTE_STRESS_IMAGE_REGISTRY": _STRESS_IMAGE_REGISTRY,
        "FLYTE_STRESS_IMAGE_NAME": _STRESS_IMAGE_NAME,
        "FLYTE_STRESS_IMAGE_PLATFORMS": ",".join(_STRESS_IMAGE_PLATFORMS),
    }.items()
    if v
}

# Let remote runs redirect image builds to a writable registry without
# touching the task definitions. For dogfood, this can point at the shared ECR
# repo used for ad hoc SDK test images. Default to amd64-only so the first
# build is faster and matches the dogfood cluster architecture.
stress_image = flyte.Image.from_debian_base(
    python_version=(3, 12),
    registry=_STRESS_IMAGE_REGISTRY,
    name=_STRESS_IMAGE_NAME,
    platform=_STRESS_IMAGE_PLATFORMS,
)


def _fanout_resources() -> flyte.Resources:
    # Default for the distributed harness shape. Override these env vars when
    # testing a single huge parent that needs much more headroom.
    cpu_request = int(os.getenv("FLYTE_STRESS_FANOUT_CPU_REQUEST", "1"))
    cpu_limit = int(os.getenv("FLYTE_STRESS_FANOUT_CPU_LIMIT", "2"))
    memory_request = os.getenv("FLYTE_STRESS_FANOUT_MEMORY_REQUEST", "2Gi")
    memory_limit = os.getenv("FLYTE_STRESS_FANOUT_MEMORY_LIMIT", "4Gi")
    return flyte.Resources(cpu=(cpu_request, cpu_limit), memory=(memory_request, memory_limit))


def _controller_tuning_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "_F_MAX_QPS",
        "_F_CTRL_WORKERS",
        "_F_P_CNC",
        "_U_USE_ACTIONS",
        "_F_TRACE_SUBMIT",
        "_F_TRACE_SUBMIT_LIMIT",
    ):
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


def _nested_run_env() -> dict[str, str]:
    return {
        **_STRESS_RUNTIME_ENV,
        **_controller_tuning_env(),
    }


def _controller_tuning_summary() -> str:
    env = _controller_tuning_env()
    return (
        "controller_env "
        f"_F_MAX_QPS={env.get('_F_MAX_QPS', '<unset>')} "
        f"_F_CTRL_WORKERS={env.get('_F_CTRL_WORKERS', '<unset>')} "
        f"_F_P_CNC={env.get('_F_P_CNC', '<unset>')} "
        f"_U_USE_ACTIONS={env.get('_U_USE_ACTIONS', '<unset>')} "
        f"_F_TRACE_SUBMIT={env.get('_F_TRACE_SUBMIT', '<unset>')} "
        f"_F_TRACE_SUBMIT_LIMIT={env.get('_F_TRACE_SUBMIT_LIMIT', '<unset>')}"
    )


# Leaves run in leaseworker via the core-sleep plugin: no task pods are created,
# so we can fan out wide without paying pod-startup cost.
sleep_env = flyte.TaskEnvironment(
    name="sleep_fanout_leaf",
    image=stress_image,
    env_vars=_STRESS_RUNTIME_ENV,
    plugin_config=Sleep(),
)

fanout_env = flyte.TaskEnvironment(
    name="sleep_fanout",
    image=stress_image,
    env_vars=_STRESS_RUNTIME_ENV,
    resources=_fanout_resources(),
    depends_on=[sleep_env],
)

swarm_env = flyte.TaskEnvironment(
    name="sleep_fanout_swarm",
    image=stress_image,
    env_vars=_STRESS_RUNTIME_ENV,
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
    print(
        f"fanout_inputs n_children={n_children} "
        f"sleep_duration={sleep_duration} "
        f"sleep_seconds={sleep_duration.total_seconds()}",
        flush=True,
    )
    print(_controller_tuning_summary(), flush=True)
    await asyncio.gather(*(sleep_leaf(duration=sleep_duration) for _ in range(n_children)))
    print(f"Done. Total leaves: {n_children}", flush=True)
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
    child_run_env = _nested_run_env()

    async def submit_one(idx: int) -> str:
        async with limiter:
            run = await flyte.run.aio(
                sleep_fanout.override(env_vars=child_run_env),
                n_children=n_children,
                sleep_duration=sleep_duration,
            )
            print(f"submitted_run idx={idx} url={run.url}", flush=True)
            return run.url

    urls = await asyncio.gather(*(submit_one(i) for i in range(n_runs)))
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
