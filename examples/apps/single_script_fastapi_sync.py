import fastapi
import uvicorn

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment fastapi-app}}
app = fastapi.FastAPI()

env = FastAPIAppEnvironment(
    name="fastapi-server-sync-example",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    port=8080,
)

@env.server
def server():
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


@app.get("/")
async def root() -> dict:
    return {"message": "Hello from FastAPI!"}
# {{/docs-fragment fastapi-app}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    deployed_app = flyte.serve(env)
    print(f"App served at: {deployed_app.url}")
# {{/docs-fragment deploy}}
