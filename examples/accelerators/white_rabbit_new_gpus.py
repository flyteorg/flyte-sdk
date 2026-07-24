"""
Targeting White_rabbit's new GPU node pools (cluster: uc-us-central1).

Two new node pools were added for White_rabbit in
deploy/dataplane/production/white_rabbit-uc-us-central1.yaml:

  | Pool                 | Machine          | GPU        | Capacity  | gpu= string   | node label value |
  |----------------------|------------------|------------|-----------|---------------|------------------|
  | white_rabbita2ultragpu1g   | a2-ultragpu-1g   | A100 80GB  | on-demand | "A100 80G:1"  | nvidia-a100-80gb |
  | white_rabbita3highgpu1gspot| a3-highgpu-1g    | H100 80GB  | spot      | "H100:1"      | nvidia-h100      |

How targeting works: the `gpu="<device>:<count>"` you pass is normalized to a node-label
value and injected as a *required* nodeAffinity on `k8s.amazonaws.com/accelerator`, plus a
matching toleration. So a task only lands on the pool whose label matches its device.

Two things specific to these pools:
  * H100 is exposed by the SDK only as "H100" (there is no "H100 80G" device); it normalizes
    to the `nvidia-h100` label. Use "A100 80G" for the 80GB A100 (distinct from the existing
    40GB "A100" pool, which is labeled nvidia-tesla-a100).
  * The H100 pool is spot, so its nodes also carry `union.ai/capacity-type=interruptible`.
    A task must set `interruptible=True` to tolerate that taint and schedule there.

Both pools are reserved (tainted on `k8s.amazonaws.com/accelerator`), so only tasks that
explicitly request the device land on them — an untyped `gpu="1"` request will NOT.

Run:  python white_rabbit_new_gpus.py
"""

import flyte

# A100 80GB, on-demand. -> node label k8s.amazonaws.com/accelerator=nvidia-a100-80gb
a100_env = flyte.TaskEnvironment(
    "white_rabbit-a100-80g",
    resources=flyte.Resources(gpu="A100 80G:1"),
)

# H100 80GB, spot. -> node label k8s.amazonaws.com/accelerator=nvidia-h100
# interruptible=True is REQUIRED: the pool is spot and tainted union.ai/capacity-type=interruptible.
h100_env = flyte.TaskEnvironment(
    "white_rabbit-h100",
    resources=flyte.Resources(gpu="H100:1"),
    interruptible=True,
)

driver_env = flyte.TaskEnvironment(
    "white_rabbit-gpu-driver",
    depends_on=[a100_env, h100_env],
)


def _gpu_name() -> str:
    """Return the GPU model visible on the node (proves which pool the task landed on)."""
    import subprocess

    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    return out.strip()


@a100_env.task
async def on_a100_80g() -> str:
    return _gpu_name()


@h100_env.task
async def on_h100() -> str:
    return _gpu_name()


@driver_env.task
async def run_on_new_gpus() -> dict[str, str]:
    import asyncio

    a100, h100 = await asyncio.gather(on_a100_80g(), on_h100())
    return {"A100 80G": a100, "H100": h100}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(run_on_new_gpus)
    print(r.url)
