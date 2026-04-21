import flyte

t4_env = flyte.TaskEnvironment("nvidia-t4", resources=flyte.Resources(gpu="T4:1"))
a10g_env = flyte.TaskEnvironment("nvidia-a10g", resources=flyte.Resources(gpu="A10G:1"))
l4_env = flyte.TaskEnvironment("nvidia-l4", resources=flyte.Resources(gpu="L4:1"))
l40s_env = flyte.TaskEnvironment("nvidia-l40s", resources=flyte.Resources(gpu="L40s:1"))
v100_env = flyte.TaskEnvironment("nvidia-v100", resources=flyte.Resources(gpu="V100:1"))

driver_env = flyte.TaskEnvironment(
    "nvidia-sweep",
    depends_on=[t4_env, a10g_env, l4_env, l40s_env, v100_env],
)


def _gpu_name() -> str:
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    )
    return out.strip()


@t4_env.task
async def on_t4() -> str:
    return _gpu_name()


@a10g_env.task
async def on_a10g() -> str:
    return _gpu_name()


@l4_env.task
async def on_l4() -> str:
    return _gpu_name()


@l40s_env.task
async def on_l40s() -> str:
    return _gpu_name()


@v100_env.task
async def on_v100() -> str:
    return _gpu_name()


@driver_env.task
async def sweep() -> dict[str, str]:
    import asyncio
    t4, a10g, l4, l40s, v100 = await asyncio.gather(
        on_t4(), on_a10g(), on_l4(), on_l40s(), on_v100()
    )
    return {"T4": t4, "A10G": a10g, "L4": l4, "L40s": l40s, "V100": v100}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(sweep)
    print(r.url)
