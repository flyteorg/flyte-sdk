import logging

import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    name="large_inline_io",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task(max_inline_io_bytes=100 * 1024)  # Limit inline I/O to 100 KiB
async def printer_task(x: str) -> str:
    print(f"Printer task received: {x}")
    return x


@env.task
async def large_inline_io() -> str:
    small = await printer_task("Hello, world!")
    print(f"Small string result: {small}")
    # Create a large string to simulate large inline I/O
    large_string = "A" * 10**6  # 1 million characters approx. 1 MiB
    print(f"Large string created with length: {len(large_string)})")
    try:
        result = await printer_task(large_string)
        print(f"Result from printer_task: {result[:50]}...")
        return result
    except flyte.errors.InlineIOMaxBytesBreached as e:
        print(f"Inline I/O limit breached: {e}")
        raise


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root(), log_level=logging.DEBUG)
    run = flyte.run(large_inline_io)
    print(run.url)
    print("Run completed.")
