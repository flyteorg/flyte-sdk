# /// script
# requires-python = "==3.13"
# dependencies = [
#    "pandas",
#    "mashumaro",
#    "botocore",
#    "pyarrow",
# ]
# ///

import flyte
from flyte import Image

image = Image.from_uv_script(__file__, name="hello", registry="ghcr.io/flyteorg")

env = flyte.TaskEnvironment(name="t1", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(t1, data="hello world")
    print(run.name)
    print(run.url)
    run.wait(run)
