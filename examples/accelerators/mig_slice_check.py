"""Checks that a task asking for one MIG slice on the RTX PRO 6000 test pool gets one.

The pool dogfood-1-awsg7e2xlargemig splits its one GPU into four 1g.24gb instances. With the MIG
strategy single, each instance is advertised as one nvidia.com/gpu, so a task asks for one GPU and
the pod template pins it to the pool. Karpenter launches the node when the first pod is pending.

    python examples/accelerators/mig_slice_check.py

The parent fans out four checks and fails unless each got a distinct MIG device.
"""

import asyncio
import os
import shutil
import subprocess

from kubernetes.client import V1Container, V1PodSpec, V1Toleration

import flyte

POOL = "dogfood-1-awsg7e2xlargemig"
SLICES = 4

image = flyte.Image.from_debian_base(name="mig-slice-check").with_pip_packages("kubernetes", "six")

# The device sets the k8s.amazonaws.com/accelerator selector and toleration. The pool has no
# k8s.amazonaws.com/gpu-partition-size label yet, so passing partition="1g.24gb" here would add a
# selector no node can satisfy and Karpenter would never launch one. Once the pool carries that
# label, the partition can be added and the nodepool selector below dropped.
mig_env = flyte.TaskEnvironment(
    name="mig-slice-check",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi", gpu=flyte.GPU("RTX PRO 6000", 1)),
    pod_template=flyte.PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            node_selector={"karpenter.sh/nodepool": POOL},
            tolerations=[
                V1Toleration(key=k, operator="Exists", effect="NoSchedule")
                for k in ("union.ai/gpu-test", "nvidia.com/gpu", "k8s.amazonaws.com/accelerator")
            ],
        ),
    ),
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
