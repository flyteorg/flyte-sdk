"""Checks that time-sliced tasks share one physical GPU.

The cluster needs a T4 node pool whose NVIDIA device plugin has a time-slicing config that
advertises each GPU as several nvidia.com/gpu replicas (at least three for this check).

    python examples/accelerators/timeslice_check.py

Set GPU_NODE_SELECTOR (for example karpenter.sh/nodepool=my-t4-pool) to pin the tasks to a pool,
and GPU_TOLERATIONS (comma separated taint keys) to tolerate the pool's taints. Both are optional.

The parent runs three tasks at once. It fails unless all three ran on one node, saw the same GPU
UUID and overlapped in time. On a node with one GPU, three overlapping GPU pods can only fit if
the GPU is shared.
"""

import asyncio
import os
import re
import shutil
import subprocess
import time

from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1EnvVarSource,
    V1ObjectFieldSelector,
    V1PodSpec,
    V1Toleration,
)

import flyte

SHARES = 3
HOLD_SECONDS = 60

image = flyte.Image.from_debian_base(name="timeslice-check").with_pip_packages("kubernetes", "six")

node_selector = None
if selector := os.environ.get("GPU_NODE_SELECTOR"):
    key, _, value = selector.partition("=")
    node_selector = {key: value}

tolerations = [
    V1Toleration(key=k.strip(), operator="Exists", effect="NoSchedule")
    for k in os.environ.get("GPU_TOLERATIONS", "").split(",")
    if k.strip()
]

# The device sets the backend's accelerator node selector. The hostname inside a pod is the pod
# name, so the node name comes from the downward API.
ts_env = flyte.TaskEnvironment(
    name="timeslice-check",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi", gpu=flyte.GPU("T4", 1)),
    pod_template=flyte.PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    env=[
                        V1EnvVar(
                            name="NODE_NAME",
                            value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(field_path="spec.nodeName")),
                        )
                    ],
                )
            ],
            node_selector=node_selector,
            tolerations=tolerations or None,
        ),
    ),
)

driver_env = flyte.TaskEnvironment(
    name="timeslice-check-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[ts_env],
)


@ts_env.task
def gpu_share(index: int) -> dict[str, str]:
    start = time.time()
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")

    # nvidia-smi is only there when the container runtime mounts the driver utilities.
    uuid = ""
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        match = re.search(r"UUID: (GPU-[0-9a-fA-F-]+)", out)
        uuid = match.group(1) if match else ""
    if not uuid and visible.startswith("GPU-"):
        uuid = visible

    # Stay on the GPU long enough for the other shares to start, so the parent can see they overlapped.
    time.sleep(HOLD_SECONDS)
    result = {
        "index": str(index),
        "node": os.environ.get("NODE_NAME", ""),
        "visible_devices": visible,
        "gpu_uuid": uuid,
        "start": str(start),
        "end": str(time.time()),
    }
    print(result)
    return result


@driver_env.task
async def main() -> list[dict[str, str]]:
    shares = await asyncio.gather(*(gpu_share.aio(i) for i in range(SHARES)))

    nodes = {s["node"] for s in shares}
    if len(nodes) != 1 or "" in nodes:
        raise RuntimeError(f"Expected all {SHARES} tasks on one node, got nodes {sorted(nodes)}: {shares}")

    uuids = {s["gpu_uuid"] for s in shares}
    if len(uuids) != 1 or "" in uuids:
        raise RuntimeError(f"Expected all {SHARES} tasks to see one GPU UUID, got {sorted(uuids)}: {shares}")

    latest_start = max(float(s["start"]) for s in shares)
    earliest_end = min(float(s["end"]) for s in shares)
    if latest_start >= earliest_end:
        raise RuntimeError(f"The {SHARES} tasks did not run at the same time, so they did not share the GPU: {shares}")

    print(f"{SHARES} tasks shared GPU {uuids.pop()} on node {nodes.pop()}")
    return list(shares)


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main).url)
