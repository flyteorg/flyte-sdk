import flyte
import flyte.extras

env = flyte.TaskEnvironment("determinism")

@env.task
async def main(n: int) -> int:
    sum = 0
    for i in range(n):
        print(f"Sleeping {i}")
        await flyte.extras.durable_sleep.aio(5)
        sum += i
    return sum


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, 5)
    print(r.url)