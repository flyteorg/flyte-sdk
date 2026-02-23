import flyte
from flyte import GPU, Resources

gpu_env = flyte.TaskEnvironment(
    name="a100_mig",
    image="ghcr.io/flyteorg/flyte:py3.11-v2.0.0b43",
    resources=Resources(
        cpu="1",
        memory="4Gi",
        gpu= GPU(device="A100 80G", quantity=1, partition="1g.10gb"),
    ),
)


@gpu_env.task
async def hello_partition_gpu() -> str:
    import asyncio

    print("Hi from the A100 partition (1g.10gb) task!")

    total_seconds = 10 * 60
    interval = 60
    for elapsed in range(0, total_seconds, interval):
        remaining = (total_seconds - elapsed) // 60
        print(f"Running... {elapsed // 60} min elapsed, {remaining} min remaining")
        await asyncio.sleep(interval)

    msg = "Done! Ran for 10 minutes on A100 partition (1g.10gb)."
    print(msg)
    return msg


regular_gpu_env = flyte.TaskEnvironment(
    name="regular_gpu",
    image=flyte.Image.from_debian_base().with_pip_packages("torch"),
    resources=Resources(
        cpu="1",
        memory="4Gi",
        gpu=1,
    ),
)


@regular_gpu_env.task
async def hello_regular_gpu() -> str:
    import torch

    import asyncio

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        msg = f"GPU: {name}, Memory: {mem:.1f} GB, CUDA: {torch.version.cuda}"
    else:
        msg = "No GPU available"
    print(msg)

    total_seconds = 10 * 60
    interval = 60
    for elapsed in range(0, total_seconds, interval):
        remaining = (total_seconds - elapsed) // 60
        print(f"Running... {elapsed // 60} min elapsed, {remaining} min remaining")
        await asyncio.sleep(interval)

    print("Done! Ran for 10 minutes.")
    return msg


cpu_env = flyte.TaskEnvironment(
    name="cpu_only",
    image="ghcr.io/flyteorg/flyte:py3.11-v2.0.0b43",
    resources=Resources(
        cpu="2",
        memory="1Gi",
    ),
)


@cpu_env.task
async def hello_cpu_and_memory() -> str:
    import os

    cpu_count = os.cpu_count()
    msg = f"Hello from CPU task! CPUs available: {cpu_count}"
    print(msg)
    return msg


@cpu_env.task
async def long_running_cpu() -> str:
    import asyncio

    total_seconds = 30 * 60
    interval = 60
    for elapsed in range(0, total_seconds, interval):
        remaining = (total_seconds - elapsed) // 60
        print(f"Running... {elapsed // 60} min elapsed, {remaining} min remaining")
        await asyncio.sleep(interval)

    msg = "Done! Ran for 30 minutes."
    print(msg)
    return msg
