"""Runs a task on one MIG slice of an RTX PRO 6000.

The cluster needs an RTX PRO 6000 node pool with MIG enabled and a layout that has 1g.24gb slices.
"""

import subprocess

import flyte

mig_env = flyte.TaskEnvironment(
    "rtx-pro-6000-mig",
    resources=flyte.Resources(gpu=flyte.GPU("RTX_PRO_6000", 1, partition="1g.24gb")),
)


@mig_env.task
def main() -> str:
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
