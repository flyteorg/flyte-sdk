import flyte
import flyte.report

env = flyte.TaskEnvironment("reports")


@env.task(report=True)
async def task1():
    # Creating new tabs
    tab = flyte.report.get_tab("Task 1")
    tab.log("<p>Task 1 HTML log</p>")
    await flyte.report.flush.aio()


@env.task(report=True)
async def task2():
    await flyte.report.replace.aio("<href='https://www.union.ai/docs/flyte/user-guide/'>Flyte docs!</a>")
    await flyte.report.flush.aio()


@env.task
async def main():
    await task1()
    await task2()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
