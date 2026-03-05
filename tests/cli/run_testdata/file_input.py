import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    name="hello_world",
)


@env.task
async def test_file(
    project: str,
    input_file: File,
) -> str:
    return f"Got input {project=}, {input_file=}"
