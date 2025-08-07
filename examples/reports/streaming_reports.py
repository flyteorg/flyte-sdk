import flyte
import flyte.report

env = flyte.TaskEnvironment(name="streaming_report")

@env.task(report=True)
async def streaming_report_task(data: str = "hello") -> str:
    await flyte.report.log.aio("<h1>Streaming Report</h1>")
    await flyte.report.flush.aio()
    await flyte.report.log.aio("<h2>Streaming Report</h2>")
    await flyte.report.flush.aio()
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(streaming_report_task, data="world")
    print(run.name)
    print(run.url)
    # The report will be available at the run URL under the "Report" tab.