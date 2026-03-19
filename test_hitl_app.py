import fastapi
from flyteplugins.hitl import env, event_app_env

import flyte
import flyte.app.extras

app = fastapi.FastAPI()


@app.get("/")
async def root() -> str:
    return "Hello, World!"


# app_env = flyte.app.extras.FastAPIAppEnvironment(
#     name="test-deploy-app-in-task-app",
#     app=app,
#     image=(
#         flyte.Image.from_debian_base(name="testing-123").with_pip_packages("fastapi", "uvicorn", "flyte==2.0.9")
#     ),
#     resources=flyte.Resources(cpu=1, memory="512Mi"),
# )

task_env = flyte.TaskEnvironment(
    name="test-deploy-app-in-task-task",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[env],
)


@task_env.task
async def test_deploy_app_in_task() -> str:
    await flyte.init_in_cluster.aio()
    app = flyte.with_servecontext().serve(event_app_env)
    # event = await hitl.new_event.aio(
    #     "integer_input_event",
    #     data_type=int,
    #     scope="run",
    #     prompt="What should I add to x?",
    # )
    # y = await event.wait.aio()
    return app.url


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    # run = flyte.with_runcontext(project="niels").run(test_deploy_app_in_task)
    # print(run.url)

    app = flyte.with_servecontext(project="niels").serve(event_app_env)
    print(app.url)
