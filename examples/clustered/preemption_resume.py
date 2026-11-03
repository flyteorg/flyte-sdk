"""
Wait out a Kueue preemption in the same attempt and resume from the checkpoint.

When the gang is submitted to a Kueue LocalQueue, higher-priority work can preempt it after it started:
Kueue suspends the JobSet and deletes its pods. By default that ends the attempt and uses one of the
task's ``retries``. With ``resume_after_preemption=True`` the attempt and the JobSet stay; the task
reports Running with "waiting for re-admission", and when Kueue re-admits the gang its pods are
recreated. ``flyte.ctx().checkpoint`` loads this attempt's own last checkpoint, so training picks up
where it stopped, and a task with ``retries=0`` survives any number of preemptions.

Other evictions (node failures, a PodsReady timeout, a deactivated Workload) still end the attempt.
``max_runtime`` keeps counting while the gang waits.

Needs a cluster with JobSet and Kueue, a LocalQueue for the task (set QUEUE below), and a Flyte
backend v2.0.54 or later.

Runbook:
  1. Start the run and wait for "saved step 5" in the rank-0 worker's log.
  2. Preempt it: submit higher-priority work to a queue that preempts lower priority, e.g. a Job
     labelled ``kueue.x-k8s.io/queue-name: <QUEUE>`` and ``kueue.x-k8s.io/priority-class: <high>``
     that needs the gang's quota.
  3. Observe: the JobSet is suspended, the task stays on attempt 1 and reports
     "gang evicted by kueue (Preempted, ...); waiting for re-admission".
  4. When the preemptor finishes, Kueue re-admits the gang: the log prints "resumed at step N"
     with N > 0, on the same attempt, and the run succeeds.

    uv run python examples/clustered/preemption_resume.py
"""

from __future__ import annotations

import flyte
from flyte._image import DIST_FOLDER, PythonWheels
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

QUEUE = "user-queue"

image = (
    flyte.Image.from_debian_base(name="preemption_resume")
    .clone(addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"))
    .with_pip_packages("torch")
)

env = ClusteredTaskEnvironment(
    name="preemption_resume_env",
    image=image,
    resources=flyte.Resources(cpu="400m", memory="1Gi"),
    replicas=2,
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    pod_template=flyte.PodTemplate(labels={"kueue.x-k8s.io/queue-name": QUEUE}),
    # The flag under test: a preemption waits for re-admission instead of using a retry.
    failure_policy=ClusterFailurePolicy(resume_after_preemption=True),
)


@env.task
async def train(steps: int = 30, step_seconds: int = 6) -> int:
    """Count to `steps`, saving a checkpoint every step; returns the step it resumed from."""
    import asyncio
    import json

    ctx = flyte.ctx()
    rank = ctx.rank or 0
    checkpoint = ctx.checkpoint
    assert checkpoint is not None  # always available inside a task

    start = 0
    restored = await checkpoint.load()
    if restored is not None:
        start = json.loads(restored.read_bytes())["step"]
    print(f"[rank {rank}] attempt={ctx.attempt_number} resumed at step {start}", flush=True)

    for step in range(start + 1, steps + 1):
        await asyncio.sleep(step_seconds)
        if rank == 0:
            await checkpoint.save(json.dumps({"step": step}).encode())
            print(f"[rank 0] saved step {step}", flush=True)
    return start


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print("Run URL:", run.url)
    run.wait()
    print("Final phase:", run.phase)
