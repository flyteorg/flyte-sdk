import flyte

env = flyte.TaskEnvironment(
    name="large_inline_io",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def printer_task(x: str) -> str:
    print(f"Printer task received: {x}")
    return x


@env.task
async def large_inline_io() -> str:
    small = await printer_task("Hello, world!")
    print(f"Small string result: {small}")
    # Create a large string to simulate large inline I/O
    large_string = "A" * 10**6  # 1 million characters
    print(f"Large string created with length: {len(large_string)})")
    result = await printer_task(large_string)
    print(f"Result from printer_task: {result[:50]}...")
    return result


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(large_inline_io)
    print(run.url)
    print("Run completed.")
