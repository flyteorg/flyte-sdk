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

@app.get("/")
async def root() -> str:
    return "Hello, World!"


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(app.url)
