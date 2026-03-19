import flyteplugins.hitl as hitl
from flyteplugins.hitl import event_app_env

import flyte

task_env = flyte.TaskEnvironment(
    name="test-deploy-app-in-task-task",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[event_app_env],
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
    flyte.init_from_config()
    run = flyte.with_runcontext(project="niels").run(test_deploy_app_in_task)
    print(run.url)


if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.with_servecontext().serve(event_app_env)
    print(f"App URL: {app.url}")
