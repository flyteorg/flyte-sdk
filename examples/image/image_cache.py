import flyte

env2 = flyte.TaskEnvironment(
    name="inner_1",
    image=flyte.Image.from_debian_base().with_pip_packages("flytekit"),
)

env3 = flyte.TaskEnvironment(
    name="inner_2",
    image=flyte.Image.from_debian_base().with_pip_packages("boto3"),
)

env1 = flyte.TaskEnvironment(
    name="outer",
    image=flyte.Image.from_debian_base().with_pip_packages("requests"),
    depends_on=[env2, env3],
)


@env3.task
async def inner_2_task() -> str:
    import boto3

    return f"Inner 2 Task Executed with boto3 version {boto3.__version__}"


@env2.task
async def inner_1_task() -> str:
    import flytekit

    return f"Inner 1 Task Executed with numpy version {flytekit.__version__}" + await inner_2_task()


@env1.task
async def outer_task() -> str:
    import requests

    inner_1_result = await inner_1_task()
    inner_2_result = await inner_2_task()
    return f"Outer Task Executed with requests version {requests.__version__}\n{inner_1_result}\n{inner_2_result}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(outer_task)
    print(run.name)
    print(run.url)
    print("Run completed successfully.")
