"""
Test: Can we parameterize Ray worker GPU/machine type via override(plugin_config=...)?

Context (Ryan Avery question):
- He has a Ray task with workers requesting GPUs (A10, L40s, etc.)
- He wants to avoid duplicating tasks/envs for each GPU variant
- override(resources=...) does NOT propagate to Ray worker pods (only affects primary container)
- override(plugin_config=...) DOES let you swap in a new RayJobConfig at invocation time

This script tests the plugin_config override pattern with different worker resource configs.
Each task sleeps so you can inspect pods in k9s.
"""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

import flyte
import flyte.remote


# -- Image --
image = (
    flyte.Image.from_debian_base(name="ray-override-test")
    .with_apt_packages("wget")
    .with_pip_packages("flyteplugins-ray", "kubernetes")
)


# -- Ray configs for different "machine types" --
def make_ray_config(
    worker_cpu: str = "2000m",
    worker_memory: str = "4Gi",
    worker_gpu: int = 0,
    replicas: int = 1,
) -> RayJobConfig:
    """Build a RayJobConfig with configurable worker resources."""
    worker_limits = {"cpu": worker_cpu, "memory": worker_memory}
    worker_requests = {"cpu": worker_cpu, "memory": worker_memory}
    if worker_gpu > 0:
        worker_limits["nvidia.com/gpu"] = str(worker_gpu)
        worker_requests["nvidia.com/gpu"] = str(worker_gpu)

    return RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="workers",
                replicas=replicas,
                pod_template=flyte.PodTemplate(
                    pod_spec=V1PodSpec(
                        containers=[
                            V1Container(
                                name="ray-worker",
                                resources=V1ResourceRequirements(
                                    limits=worker_limits,
                                    requests=worker_requests,
                                ),
                            )
                        ],
                    )
                ),
            )
        ],
        head_node_config=HeadNodeConfig(
            ray_start_params={"num-cpus": "0", "num-gpus": "0"},
            pod_template=flyte.PodTemplate(
                pod_spec=V1PodSpec(
                    containers=[
                        V1Container(
                            name="ray-head",
                            resources=V1ResourceRequirements(
                                limits={"cpu": "2000m", "memory": "4Gi"},
                                requests={"cpu": "2000m", "memory": "4Gi"},
                            ),
                        )
                    ],
                )
            ),
        ),
        shutdown_after_job_finishes=True,
        ttl_seconds_after_finished=300,
    )


# Default config: CPU-only workers
default_ray_config = make_ray_config(worker_cpu="2000m", worker_memory="4Gi", worker_gpu=0, replicas=2)

# -- Environments --
ray_env = flyte.TaskEnvironment(
    name="ray_default",
    plugin_config=default_ray_config,
    image=image,
    resources=flyte.Resources(cpu=(2, 4), memory=("2Gi", "4Gi")),
)

cpu_env = flyte.TaskEnvironment(
    name="cpu_task",
    resources=flyte.Resources(cpu=(1, 2), memory=("400Mi", "1000Mi")),
    image=image,
    depends_on=[ray_env],
)


# -- Ray remote function --
@ray.remote
def do_work(task_id: int, sleep_seconds: int = 30) -> str:
    """Simulates work on a Ray worker. Sleeps so you can inspect in k9s."""
    import socket

    hostname = socket.gethostname()
    print(f"[Worker {task_id}] Starting on {hostname}, sleeping {sleep_seconds}s...")
    time.sleep(sleep_seconds)
    print(f"[Worker {task_id}] Done on {hostname}")
    return f"task={task_id}, host={hostname}"


# -- Tasks --
@ray_env.task
async def ray_work(n_tasks: int = 4, sleep_seconds: int = 60) -> List[str]:
    """
    Runs work on Ray workers. Sleep is long enough to inspect pods in k9s.
    Check worker pod resource requests/limits to verify the config took effect.
    """
    print(f"Dispatching {n_tasks} tasks to Ray workers, each sleeping {sleep_seconds}s")
    futures = [do_work.remote(i, sleep_seconds) for i in range(n_tasks)]
    results = ray.get(futures)
    print(f"All tasks complete: {results}")
    return results


@cpu_env.task
async def run_with_override(
    machine_type: str = "cpu",
    n_tasks: int = 4,
    sleep_seconds: int = 60,
) -> List[str]:
    """
    Demonstrates overriding plugin_config to change Ray worker resources at invocation time.

    machine_type options:
      - "cpu": 2 CPU workers, no GPU
      - "gpu_1": 2 workers, each with 1 GPU
      - "gpu_4": 1 worker with 4 GPUs

    After launching, use k9s to inspect the Ray worker pods and verify
    the resource requests/limits match the selected machine type.
    """
    configs = {
        "cpu": make_ray_config(worker_cpu="2000m", worker_memory="4Gi", worker_gpu=0, replicas=2),
        "gpu_1": make_ray_config(worker_cpu="4000m", worker_memory="16Gi", worker_gpu=1, replicas=2),
        "gpu_4": make_ray_config(worker_cpu="16000m", worker_memory="64Gi", worker_gpu=4, replicas=1),
    }
    cfg = configs.get(machine_type)
    if cfg is None:
        raise ValueError(f"Unknown machine_type: {machine_type}. Choose from: {list(configs.keys())}")

    print(f"Running Ray task with machine_type={machine_type}")
    return await ray_work.override(plugin_config=cfg)(n_tasks=n_tasks, sleep_seconds=sleep_seconds)


@cpu_env.task
async def run_comparison(sleep_seconds: int = 60) -> dict:
    """
    Runs the same Ray task twice with different configs to prove the override works.
    Watch in k9s: the first run should have CPU-only workers, the second should request GPUs.
    """
    print("=== Run 1: CPU-only workers ===")
    cpu_results = await ray_work.override(
        plugin_config=make_ray_config(worker_cpu="2000m", worker_memory="4Gi", worker_gpu=0, replicas=2)
    )(n_tasks=2, sleep_seconds=sleep_seconds)

    print("=== Run 2: GPU workers (1 GPU each) ===")
    gpu_results = await ray_work.override(
        plugin_config=make_ray_config(worker_cpu="4000m", worker_memory="16Gi", worker_gpu=1, replicas=2)
    )(n_tasks=2, sleep_seconds=sleep_seconds)

    return {"cpu_run": cpu_results, "gpu_run": gpu_results}


if __name__ == "__main__":
    flyte.init_from_config()

    # Option 1: Run with a specific machine type override
    print("Launching ray task with plugin_config override...")
    run = flyte.run(run_with_override, machine_type="gpu_1", sleep_seconds=60)
    print(f"Run name: {run.name}")
    print(f"Run URL:  {run.url}")
    print("Go check k9s! Worker pods should reflect the selected machine_type resources.")
    run.wait()

    # Option 2: Run comparison (uncomment to use instead)
    # run = flyte.run(run_comparison, sleep_seconds=60)
    # print(f"Run name: {run.name}")
    # print(f"Run URL:  {run.url}")
    # run.wait()
