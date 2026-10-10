"""Checks that a task asking for one RTX PRO 6000 MIG slice gets one.

The cluster needs the NVIDIA GPU Operator with the MIG strategy single, and a node pool whose
nodes are labeled with the MIG layout (for example nvidia.com/mig.config=all-1g.24gb) and with
k8s.amazonaws.com/gpu-partition-size=1g.24gb to match the partition requested below. With the
strategy single, each MIG instance is advertised as one nvidia.com/gpu.

    python examples/accelerators/mig_slice_check.py

Set GPU_NODE_SELECTOR (for example karpenter.sh/nodepool=my-mig-pool) to pin the tasks to a pool,
and GPU_TOLERATIONS (comma separated taint keys) to tolerate the pool's taints. Both are optional.

The parent fans out four checks and fails unless each got a distinct MIG device.
"""

import asyncio
import os
import shutil
import subprocess

from kubernetes.client import V1Container, V1PodSpec, V1Toleration

import flyte

SLICES = 4

image = flyte.Image.from_debian_base(name="mig-slice-check").with_pip_packages("kubernetes", "six")

node_selector = None
if selector := os.environ.get("GPU_NODE_SELECTOR"):
    key, _, value = selector.partition("=")
    node_selector = {key: value}

tolerations = [
    V1Toleration(key=k.strip(), operator="Exists", effect="NoSchedule")
    for k in os.environ.get("GPU_TOLERATIONS", "").split(",")
    if k.strip()
]

# The device and partition set the backend's accelerator and partition node selectors, so the pod
# template is only needed when a node pool or tolerations are given.
mig_env = flyte.TaskEnvironment(
    name="mig-slice-check",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi", gpu=flyte.GPU("RTX PRO 6000", 1, partition="1g.24gb")),
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

driver_env = flyte.TaskEnvironment(
    name="mig-slice-check-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[mig_env],
)


@mig_env.task
def check_mig_slice(index: int) -> str:
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if not visible.startswith("MIG-"):
        raise RuntimeError(
            f"Check {index} expected a MIG device but NVIDIA_VISIBLE_DEVICES is {visible!r}. "
            "If it starts with GPU-, the pod got the whole GPU, which happens when it starts before "
            "the MIG manager has split the GPU."
        )

    # nvidia-smi is only there when the container runtime mounts the driver utilities.
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        migs = [line.strip() for line in out.splitlines() if line.strip().startswith("MIG ")]
        if len(migs) != 1:
            raise RuntimeError(f"Check {index} expected exactly one MIG device in nvidia-smi -L, got:\n{out}")

    print(f"Check {index} got {visible}")
    return visible


@driver_env.task
async def main() -> list[str]:
    uuids = await asyncio.gather(*(check_mig_slice.aio(i) for i in range(SLICES)))
    if len(set(uuids)) != SLICES:
        raise RuntimeError(f"Expected {SLICES} distinct MIG devices, got {uuids}")
    return list(uuids)


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main).url)
