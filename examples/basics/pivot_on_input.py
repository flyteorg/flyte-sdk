import asyncio
import flyte
import flyte.git


env = flyte.TaskEnvironment("pivot", resources=flyte.Resources(cpu="1"))


@env.task
async def driver(cpu_inputs: list[int]) -> list[str]:
    vals = []
    with flyte.group("pivot-on-cpu"):
        for cpu in cpu_inputs:
            vals.append(pivot.override(resources=flyte.Resources(cpu=cpu))(cpu))

        results = await asyncio.gather(*vals)
        return results


@env.task
async def pivot(input: int) -> str:
    print (f"Running with {input} cpus...")
    return f"Done-{input}"


if __name__ == "__main__":
    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(driver, cpu_inputs=[1,2,4])
    print(run.url)


