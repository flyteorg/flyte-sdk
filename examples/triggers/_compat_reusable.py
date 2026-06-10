"""Backward-compat test: reusable (actor) task with an inline-bound minutely trigger.

Pins a PUBLISHED flyte base (no with_local_v2) so the actor runs the released SDK, plus the only
unionai-reuse worker that targets flyte>=2.3.6 (0.1.15b0). Uses minutely("start_time") so the kickoff
arg is bound (inline-valid for non-offloading SDKs).

FLYTE_VER env var selects the base flyte version.
"""

import os
from datetime import datetime

import flyte

_ver = os.environ.get("FLYTE_VER", "2.4.4")
_reuse = os.environ.get("REUSE_VER", "0.1.15b0")
image = flyte.Image.from_debian_base(flyte_version=_ver).with_pip_packages(
    f"unionai-reuse=={_reuse}",
    extra_index_urls="https://test.pypi.org/simple/",
    pre=True,
)

env = flyte.TaskEnvironment(
    name=os.environ.get("COMPAT_ENV", "compat_reusable"),
    image=image,
    resources=flyte.Resources(cpu="1", memory="500Mi"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=2),
    env_vars={"_U_USE_ACTIONS": "1"},
)


@env.task(triggers=flyte.Trigger.minutely("start_time"))
async def compat_reusable_task(start_time: datetime, x: int = 9) -> str:
    return f"compat_reusable executed at {start_time.isoformat()} with x={x}"


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
