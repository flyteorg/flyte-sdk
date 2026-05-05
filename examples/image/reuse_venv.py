import flyte
from flyte import Image

# Dockerfile.reuse_venv creates a custom venv at /app/venv
# and pre-installs packages (e.g. requests). We reuse that image as a Flyte base image.
image = (
    Image.from_base("ghcr.io/flyteorg/reuse-venv-example:latest")
    .clone(name="flyte", extendable=True)
    .with_env_vars({"UV_PYTHON": "/app/venv/bin/python"})
    .with_pip_packages("flyte")
)
env = flyte.TaskEnvironment(name="reuse_venv", image=image)


@env.task
async def t1(url: str = "https://httpbin.org/get") -> str:
    import requests

    resp = requests.get(url)
    return resp.text


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(t1, url="https://httpbin.org/get")
    print(run.name)
    print(run.url)
    run.wait()
