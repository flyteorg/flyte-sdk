from typing import Optional

import flyte

env = flyte.TaskEnvironment(
    name="traces",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


@flyte.trace
async def square(x: int) -> int:
    print(f"Square {x}", flush=True)
    return x**2


@flyte.trace
async def print_now(x: int, y: int):
    print(f"Now {x} {y}", flush=True)
    print(f"x: {x}, y: {y}", flush=True)


@flyte.trace
async def no_inputs() -> int:
    print(f"Now {1}", flush=True)
    return 42


@flyte.trace
async def no_inputs_outputs():
    print(f"Now {2}", flush=True)
    print("Hello World", flush=True)


@flyte.trace
async def get_optional() -> Optional[str]:
    return None


@flyte.trace
async def echo_optional(val: Optional[str]) -> Optional[str]:
    return val


@flyte.trace
async def get_bool() -> bool:
    return False


@env.task
async def traces(n: int) -> int:
    x = await square(n)
    await print_now(n, x)
    no_input_val = await no_inputs()
    print(f"No input val: {no_input_val}", flush=True)
    await no_inputs_outputs()
    return x


@env.task
async def traces_loop(n: int = 3) -> int:
    total = 0
    for i in range(n):
        sq = await square(i)
        total += sq
    return total


@env.task
async def traces_complex(n: int = 3) -> int:
    total = 0
    for i in range(n):
        sq = await square(i)
        await print_now(i, sq)
        total += sq

    no_input_val = await no_inputs()
    print(f"No input val: {no_input_val}", flush=True)
    await no_inputs_outputs()
    opt_val = await get_optional()
    print(f"Optional val: {opt_val}", flush=True)
    echoed_val = await echo_optional(opt_val)
    print(f"Echoed Optional val: {echoed_val}", flush=True)
    bool_val = await get_bool()
    print(f"Bool val: {bool_val}", flush=True)
    return total


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext().run(traces_complex, n=5)
    print(run.name)
    print(run.url)
    run.wait()
