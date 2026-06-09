"""
Trigger example using a reusable TaskEnvironment.

Mirrors basic.py, but the environment keeps a warm pool of containers (ReusePolicy) so scheduled
runs reuse already-started replicas instead of cold-starting on every fire. Tasks are `async`
because the policy uses concurrency > 1, which is only supported for async tasks.

Reusable containers require the `unionai-reuse` runtime library in the *task image* (it provides
the `unionai-actor-bridge` the actor pod execs); it does not need to be installed locally. Here we
install the GATED worker prerelease from TestPyPI (the actor run_start_time changes) and layer the
local flyte SDK on top with `with_local_v2`. Reusable containers only work against a Union backend.
"""

from datetime import datetime

import flyte

image = (
    # Pin the base to a PUBLISHED flyte release (v2.4.4 is latest on ghcr) rather than letting it track
    # the local SDK version (2.4.5, which has no published base image). with_local_v2() below overlays
    # our 2.4.5 wheel on top, so the actual flyte code in the image is still your local rusty change.
    flyte.Image.from_debian_base(flyte_version="2.4.4")
    # GATED worker prerelease (capability-probe actor_environment.rs + task_args.rs run_start_time
    # threading), published to TestPyPI. extra_index_urls keeps PyPI primary so deps still resolve;
    # pre=True allows the b0 prerelease.
    .with_pip_packages(
        "unionai-reuse==0.1.15b0",
        extra_index_urls="https://test.pypi.org/simple/",
        pre=True,
    )
    .with_local_v2()  # layer the local flyte SDK wheel (rusty.py run_start_time: Optional[str]) from ./dist
)

env = flyte.TaskEnvironment(
    name="reusable_trigger_example",
    image=image,
    resources=flyte.Resources(cpu="1", memory="500Mi"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=2),
    env_vars={"_U_USE_ACTIONS": "1"},
)


@env.task(triggers=flyte.Trigger.hourly())  # Every hour
async def reusable_example_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Reusable task executed at {trigger_time.isoformat()} with x={x}"


custom_trigger = flyte.Trigger(
    "custom_cron",
    flyte.Cron("0 0 * * *", timezone="Europe/London"),  # Runs once every day at midnight London time
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)


@env.task(triggers=(custom_trigger, flyte.Trigger.minutely("start_time")))  # Custom trigger and every minute
async def reusable_custom_task(start_time: datetime, x: int = 11) -> str:
    return f"Reusable custom task executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
