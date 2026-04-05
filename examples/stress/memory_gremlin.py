"""Allocates 10 MB every second for 1 minute to stress memory limits."""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="memory_gremlin")


@env.task
async def memory_gremlin() -> str:
    """Allocate 10 MB every second for 60 seconds, holding all allocations."""
    allocations = []
    for i in range(60):
        chunk = bytearray(10 * 1024 * 1024)  # 10 MB
        allocations.append(chunk)
        total_mb = len(allocations) * 10
        print(f"[{i + 1}/60] Allocated 10 MB — total held: {total_mb} MB")
        await asyncio.sleep(1)
    return f"Done. Held {len(allocations) * 10} MB total."


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(memory_gremlin)
    print(r.url)
