import fastapi
import uvicorn

import flyte
import flyte.app

env = flyte.app.AppEnvironment(
    name="pickled-app",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    inputs=[flyte.app.Input(name="foo", value="bar")],
    requires_auth=False,
    port=8080,
)


@env.server
def fastapi_app_startup(foo: str):
    # NOTE: since FastAPI cannot be pickled (because starlette.datastructures.State cannot be pickled due to
    # circular references), we need to use a factory pattern to create the app instance. In the startup function.
    app = fastapi.FastAPI()

    @app.get("/")
    async def root() -> str:
        return f"Hello, World! {foo}"

    uvicorn.run(app, port=8080)


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(app.url)
