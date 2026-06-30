import fastapi

import flyte
import flyte.app.extras

app = fastapi.FastAPI()


@app.get("/")
async def root() -> str:
    return "Hello, World!"


app_env = flyte.app.extras.FastAPIAppEnvironment(
    name="test-deploy-app-in-task-app",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "flyte==2.0.9"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

task_env = flyte.TaskEnvironment(
    name="test-deploy-app-in-task-task",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[app_env],
)


@task_env.task
async def test_deploy_app_in_task() -> str:
    await flyte.init_in_cluster.aio()
    app = flyte.with_servecontext(interactive_mode=False).serve(app_env)
    return app.url


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(project="niels").run(test_deploy_app_in_task)
    print(run.url)
