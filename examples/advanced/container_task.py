import flyte
from flyte.extras import ContainerTask

env = flyte.TaskEnvironment(name="hello_world")


greeting_task = ContainerTask(
    name="echo_and_return_greeting",
    image="alpine:latest",
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs={"name": str},
    outputs={"greeting": str},
    command=["/bin/sh", "-c", "echo 'Hello, my name is {{.inputs.name}}.' | tee -a /var/outputs/greeting"],
)

# Add it to the environment if you want to deploy it.
env.add_task(greeting_task)


@env.task
async def say_hello(name: str = "flyte") -> str:
    print("Hello container task")
    return await greeting_task(name=name)


if __name__ == "__main__":
    # asyncio.run(say_hello("Union"))
    import flyte.storage

    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext(mode="remote").run(say_hello, "Union")
    print(run.url)
