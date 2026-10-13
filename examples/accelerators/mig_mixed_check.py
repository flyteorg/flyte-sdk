"""Checks that tasks asking for different RTX PRO 6000 MIG profiles share one GPU.

The cluster needs the NVIDIA GPU Operator with the MIG strategy mixed, and a node pool whose
nodes are labeled with the all-balanced layout (nvidia.com/mig.config=all-balanced), which splits
each RTX PRO 6000 into two 1g.24gb instances and one 2g.48gb instance. With the strategy mixed,
the device plugin advertises each profile as its own resource, such as nvidia.com/mig-1g.24gb,
and the backend requests that resource for a task that asks for a partition, so the pool needs no
partition size label.

    python examples/accelerators/mig_mixed_check.py

Set GPU_NODE_SELECTOR (for example karpenter.sh/nodepool=my-mig-pool) to pin the tasks to a pool,
and GPU_TOLERATIONS (comma separated taint keys) to tolerate the pool's taints. Set both or
neither. The example stops before launching when only one of them is set.

The parent fans out two 1g.24gb checks and one 2g.48gb check, which together fill one GPU, and
fails unless each got one MIG device of the profile it asked for and all three are distinct.
"""

import asyncio
import os
import re
import shutil
import subprocess
import time
from typing import Literal

from kubernetes.client import V1Container, V1PodSpec, V1Toleration

import flyte

HOLD_SECONDS = 60

image = flyte.Image.from_debian_base(name="mig-mixed-check").with_pip_packages("kubernetes", "six")

# The driver task builds the child task specs inside its own container when it fans them out, so
# the module is imported again there. Both variables are passed on to the containers through
# env_vars, otherwise the child tasks would lose the node selector and the tolerations.
GPU_NODE_SELECTOR = os.environ.get("GPU_NODE_SELECTOR", "")
GPU_TOLERATIONS = os.environ.get("GPU_TOLERATIONS", "")
if bool(GPU_NODE_SELECTOR) != bool(GPU_TOLERATIONS):
    raise ValueError("Set both GPU_NODE_SELECTOR and GPU_TOLERATIONS, or neither of them.")
if GPU_NODE_SELECTOR and "=" not in GPU_NODE_SELECTOR:
    raise ValueError(f"GPU_NODE_SELECTOR must look like key=value, got {GPU_NODE_SELECTOR!r}.")

carried_env_vars = (
    {"GPU_NODE_SELECTOR": GPU_NODE_SELECTOR, "GPU_TOLERATIONS": GPU_TOLERATIONS} if GPU_NODE_SELECTOR else None
)

node_selector = None
if GPU_NODE_SELECTOR:
    key, _, value = GPU_NODE_SELECTOR.partition("=")
    node_selector = {key: value}

tolerations = [
    V1Toleration(key=k.strip(), operator="Exists", effect="NoSchedule") for k in GPU_TOLERATIONS.split(",") if k.strip()
]


def _mig_env(name: str, partition: Literal["1g.24gb", "2g.48gb"]) -> flyte.TaskEnvironment:
    # The device and partition set the backend's accelerator node selector and the per-profile
    # resource, so the pod template is only needed when a node pool or tolerations are given.
    return flyte.TaskEnvironment(
        name=name,
        image=image,
        env_vars=carried_env_vars,
        resources=flyte.Resources(cpu=1, memory="2Gi", gpu=flyte.GPU("RTX_PRO_6000", 1, partition=partition)),
        pod_template=flyte.PodTemplate(
            pod_spec=V1PodSpec(
                containers=[V1Container(name="primary")],
                node_selector=node_selector,
                tolerations=tolerations or None,
            ),
        )
        if node_selector or tolerations
        else None,
    )


mig_1g_env = _mig_env("mig-mixed-check-1g", "1g.24gb")
mig_2g_env = _mig_env("mig-mixed-check-2g", "2g.48gb")

driver_env = flyte.TaskEnvironment(
    name="mig-mixed-check-driver",
    image=image,
    env_vars=carried_env_vars,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[mig_1g_env, mig_2g_env],
)


def _check(profile: str, index: int) -> str:
    # nvidia-smi -L lists a MIG device as "MIG 2g.48gb     Device  0: (UUID: MIG-...)", with the
    # columns padded by spaces. It is only there when the container runtime mounts the driver
    # utilities.
    if not shutil.which("nvidia-smi"):
        raise RuntimeError(f"{profile} check {index} needs nvidia-smi in the container to read its MIG device.")
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    migs = re.findall(r"MIG (\S+)\s+Device\s+\d+: \(UUID: (MIG-[^)]+)\)", out)
    if len(migs) != 1 or migs[0][0] != profile:
        raise RuntimeError(f"{profile} check {index} expected exactly one {profile} MIG device, got:\n{out}")

    # Hold the slice for a while so GPU metrics scrapes see the pod on it.
    time.sleep(HOLD_SECONDS)
    print(f"{profile} check {index} got {migs[0][1]}")
    return migs[0][1]


@mig_1g_env.task
def check_1g(index: int) -> str:
    return _check("1g.24gb", index)


@mig_2g_env.task
def check_2g(index: int) -> str:
    return _check("2g.48gb", index)


@driver_env.task
async def main() -> list[str]:
    uuids = await asyncio.gather(check_1g.aio(0), check_1g.aio(1), check_2g.aio(0))
    if len(set(uuids)) != len(uuids):
        raise RuntimeError(f"Expected {len(uuids)} distinct MIG devices, got {uuids}")
    return list(uuids)


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main).url)
