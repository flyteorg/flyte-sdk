import flyte
from flyte.extras import ContainerTask

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse").with_local_v2()
env = flyte.TaskEnvironment(name="reusable_container_task_env")

reusable_container_task = ContainerTask(
    name="reusable_container_task",
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=(1, 4),
        idle_ttl=300,
    ),
    inputs={"name": str},
    command=["/bin/sh", "-c", "echo 'Hello, my name is {{.inputs.name}}.'"],
)

env.add_task(reusable_container_task)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.with_runcontext(mode="remote").run(reusable_container_task, "Union")
    print(run.url)
