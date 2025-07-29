import pathlib

import flyte
from flyte.extras import ContainerTask
from flyte.io import File

env = flyte.TaskEnvironment(name="hello_world")


@env.task
async def build_run_code() -> int:
    print("Hello container task")
    greeting_task = ContainerTask(
        name="echo_and_return_greeting",
        image="ghcr.io/astral-sh/uv:debian-slim",
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={"script.py": File, "a": int, "b": int},  # Note the script.py name here
        outputs={"result": int},
        command=["/bin/sh", "-c", "uv run /var/inputs/script.py {{.inputs.a}} {{.inputs.b}} > /var/outputs/result"],
    )
    path = pathlib.Path(__file__).parent / "llm_generated.py"
    f = await File.from_local(path)
    kwargs = {"script.py": f, "a": 1, "b": 2}
    return await greeting_task(**kwargs)


@env.task
async def build_run_code2() -> int:
    print("Hello container task")
    greeting_task = ContainerTask(
        name="echo_and_return_greeting",
        image="ghcr.io/astral-sh/uv:debian-slim",
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={"script": File, "a": int, "b": int},
        outputs={"result": int},
        command=[
            "/bin/sh",
            "-c",
            "uv run --script /var/inputs/script {{.inputs.a}} {{.inputs.b}} > /var/outputs/result",
        ],
    )
    path = pathlib.Path(__file__).parent / "llm_generated.py"
    f = await File.from_local(path)
    return await greeting_task(script=f, a=1, b=2)


if __name__ == "__main__":
    # NOTE the root_dir is set to the current directory so that the "copy_style=all" only copies the parent directory
    # and all files in it, including the script.
    flyte.init_from_config("../../config.yaml", root_dir=pathlib.Path(__file__).parent)
    r = flyte.with_runcontext(copy_style="all").run(build_run_code2)  # Copy all files, including the script
    print(r.url)
