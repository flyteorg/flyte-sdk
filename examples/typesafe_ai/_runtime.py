"""Shared Flyte runtime environments (image + all demo secrets).

Two environments, differing only in size:

``env``
    The per-unit worker. One pipeline run at a time, so 1 CPU / 1 GiB is plenty.

``driver_env``
    The benchmark driver. It holds every ``UnitResult`` in memory at once plus
    the runtime's bookkeeping for every sub-action it launched, then renders the
    whole report — so its footprint grows with the size of the matrix, not with
    the work of any single unit. A 1,440-unit run OOM-killed the driver at 1 GiB
    (run ``uw7xsx827dhgr8vprnss``, 24 minutes in, losing the entire run), hence
    the separate, larger environment.
"""

from _config import ANTHROPIC_SECRET, QWEN_SECRET, TYPESAFE_SECRET

import flyte

_IMAGE = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx", "typesafe-sdk")

# One environment carries every secret the demo needs so a single pipeline can
# interleave Jev (System 1) with any System 2 provider (Qwen / Claude).
env = flyte.TaskEnvironment(
    name="typesafe-ai",
    secrets=[
        flyte.Secret(TYPESAFE_SECRET, as_env_var=TYPESAFE_SECRET),
        flyte.Secret(QWEN_SECRET, as_env_var=QWEN_SECRET),
        flyte.Secret(ANTHROPIC_SECRET, as_env_var=ANTHROPIC_SECRET),
    ],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=_IMAGE,
)

# The aggregator/reporter. Scales with the number of units in the matrix.
driver_env = flyte.TaskEnvironment(
    name="typesafe-ai-driver",
    secrets=[
        flyte.Secret(TYPESAFE_SECRET, as_env_var=TYPESAFE_SECRET),
        flyte.Secret(QWEN_SECRET, as_env_var=QWEN_SECRET),
        flyte.Secret(ANTHROPIC_SECRET, as_env_var=ANTHROPIC_SECRET),
    ],
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    image=_IMAGE,
    # The driver fans out into `env`; a cross-environment call has to declare the
    # dependency or the worker environment is missing from the image cache at
    # runtime ("Environment 'typesafe-ai' not found in image cache").
    depends_on=[env],
)
