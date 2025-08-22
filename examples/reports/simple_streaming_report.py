import asyncio

import flyte
import flyte.report

env = flyte.TaskEnvironment(name="simple_streaming_reports")

@env.task(report=True)
async def stream_report(epochs: int = 60) -> str:
    await flyte.report.log.aio("<h1>Simple streaming report</h1>", do_flush=True)
    print(f"Simple streaming report started for {epochs} epochs.", flush=True)

    for epoch in range(1, epochs + 1):
        await flyte.report.log.aio("<p>x</p>", do_flush=True)
        await asyncio.sleep(1)  # Update every second

    await flyte.report.log.aio("Done.", do_flush=True)
    print("Simple streaming report completed", flush=True)
    return "Simple streaming report completed"


@env.task
async def main():
    await stream_report(epochs=60)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(main)
    print(f"Run Name: {run.name}", flush=True)
    print(f"Run URL: {run.url}", flush=True)
