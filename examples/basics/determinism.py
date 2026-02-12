import asyncio

import flyte
import flyte.durable

env = flyte.TaskEnvironment("determinism")


@env.task
async def async_variant():
    t = await flyte.durable.time.aio()
    dt = await flyte.durable.now.aio()
    print(f"[async] time={t}, now={dt}")
    await flyte.durable.sleep.aio(1)
    print("[async] done sleeping")


@env.task
def sync_variant():
    t = flyte.durable.time()
    dt = flyte.durable.now()
    print(f"[sync] time={t}, now={dt}")
    flyte.durable.sleep(1)
    print("[sync] done sleeping")


@env.task
async def main():
    await asyncio.gather(async_variant(), sync_variant.aio())


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
