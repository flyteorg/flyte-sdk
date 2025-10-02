import flyte
from flyte.extras import ContainerTask


def test_from_task_sets_env():
    greeting_task = ContainerTask(
        name="echo_and_return_greeting",
        image=flyte.Image.from_base("alpine:3.18"),
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={"name": str},
        outputs={"greeting": str},
        command=["/bin/sh", "-c", "echo 'Hello, my name is {{.inputs.name}}.' | tee -a /var/outputs/greeting"],
    )

    flyte.TaskEnvironment.from_task("container_env", greeting_task)

    assert greeting_task.parent_env_name == "container_env"
