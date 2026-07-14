import flyte
from flyte.extras import ContainerTask

greeting_task = ContainerTask(
    name="echo_and_return_greeting",
    image=flyte.Image.from_base("alpine:3.18"),
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs={"name": str},
    outputs={"greeting": str},
    command=["/bin/sh", "-c", "echo 'Hello, my name is {{.inputs.name}}.' | tee -a /var/outputs/greeting"],
)

container_env = flyte.TaskEnvironment.from_task("container_env", greeting_task)

env = flyte.TaskEnvironment(name="hello_world", depends_on=[container_env])


@env.task
async def say_hello(name: str = "flyte") -> str:
    print("Hello container task")
    return await greeting_task(name=name)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(say_hello, "Union")
    print(run.url)
