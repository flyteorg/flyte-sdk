import flyte
from kubernetes.client import V1Container, V1PodSpec, V1Toleration

g6_test_template = flyte.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary")],
        tolerations=[
            V1Toleration(key="nvidia.com/gpu", operator="Equal", value="present", effect="NoSchedule"),
            V1Toleration(key="union.ai/pool", operator="Equal", value="gpu-g6-test", effect="NoSchedule"),
        ],
    ),
)

env = flyte.TaskEnvironment(
    name="g6-test",
    resources=flyte.Resources(gpu=flyte.GPU(device="L4", quantity=1)),
    pod_template=g6_test_template,
)


@env.task
async def my_gpu_task() -> str:
    import asyncio
    print("Running on g6.4xlarge with NVIDIA L4", flush=True)
    await asyncio.sleep(20)
    return "Running on g6.4xlarge with NVIDIA L4"