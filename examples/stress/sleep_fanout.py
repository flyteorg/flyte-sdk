import asyncio
from datetime import timedelta

from flyteplugins.sleep import Sleep

import flyte

# Build the flyteplugins-sleep wheel locally before submitting:
#   make dist && FLYTE_PLUGIN_DIST=plugins/sleep make dist-plugins
# `with_local_v2_plugins` then installs it into the image from `flyte-sdk/dist/`.
image = flyte.Image.from_debian_base().with_local_v2_plugins("flyteplugins-sleep")

# Leaves run in leaseworker via the core-sleep plugin: no task pods are created,
# so we can fan out wide without paying pod-startup cost.
sleep_env = flyte.TaskEnvironment(
    name="sleep_fanout_leaf",
    image=image,
    plugin_config=Sleep(),
)

# Parent runs as a normal container task; it needs flyteplugins-sleep installed
# because the module-level `from flyteplugins.sleep import Sleep` runs on load.
fanout_env = flyte.TaskEnvironment(
    name="sleep_fanout",
    image=image,
    resources=flyte.Resources(cpu="50m", memory="200Mi"),
    depends_on=[sleep_env],
)


@sleep_env.task
async def sleep_leaf(duration: timedelta) -> None:
    return None


@fanout_env.task
async def main(
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


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext("remote").run(
        main,
        n_children=10,
        sleep_duration=timedelta(seconds=30),
    )
    print(run.url)
