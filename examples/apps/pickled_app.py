import fastapi
import uvicorn

import flyte
import flyte.app

env = flyte.app.AppEnvironment(
    name="pickled-app",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    parameters=[flyte.app.Parameter(name="foo", value="bar")],
    requires_auth=False,
    port=8080,
)

state = {}


@env.on_startup
async def app_startup(foo: str):
    # add state: this is where you might load data, models, open connections, etc.
    print("Starting up app...")
    state["bar"] = foo


@env.server
async def app_server(foo: str):
    # NOTE: since FastAPI cannot be pickled (because starlette.datastructures.State cannot be pickled due to
    # circular references), we need to use a factory pattern to create the app instance. In the startup function.
    app = fastapi.FastAPI()

    @app.get("/")
    async def root() -> str:
        return f"Hello, World! {foo}, here is the state: {state}"

    print("Starting up uvicorn server...")
    await uvicorn.Server(uvicorn.Config(app, port=8080)).serve()


@env.on_shutdown
async def app_shutdown(**kwargs):
    # clear the state on shutdown
    print("Shutting down app, clearing state...")
    state.clear()


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app = flyte.with_servecontext(
        interactive_mode=True,
        log_level=logging.DEBUG,
    ).serve(env)
    print(app.url)
