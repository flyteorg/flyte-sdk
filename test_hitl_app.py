import flyteplugins.hitl as hitl
from flyteplugins.hitl import env

import flyte

task_env = flyte.TaskEnvironment(
    name="test-deploy-app-in-task-task",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[env],
)


@task_env.task
async def test_deploy_app_in_task() -> int:
    event = await hitl.new_event.aio(
        "integer_input_event",
        data_type=int,
        scope="run",
        prompt="What should I add to x?",
    )
    y = await event.wait.aio()
    return y


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(project="niels").run(test_deploy_app_in_task)
    print(run.url)
