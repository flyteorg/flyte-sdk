import fastapi

import flyte
from flyte.app.extras import FastAPIAppEnvironment

app = fastapi.FastAPI()

env = FastAPIAppEnvironment(
    name="pickled-fastapi-app",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    port=8080,
)

state = {}


@env.on_startup
async def app_startup():
    state["foo"] = "bar"


@app.get("/")
async def root() -> str:
    return f"Hello, World! Here's the state: {state}"


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app = flyte.with_servecontext(interactive_mode=False).serve(env)
    print(app.url)
