import logging
import pathlib

from my_task_library import flyte_entities

import flyte

img = flyte.Image.from_debian_base(install_flyte=False).with_pip_packages("my-task-library").with_local_v2()
env = flyte.TaskEnvironment(
    name="library-consumer-env",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=img,
    depends_on=[flyte_entities.library_environment],
)


@env.task
async def use_library(data: str = "default string", n: int = 3) -> str:
    result = await flyte_entities.library_parent_task(data, n)
    print(f"Result from library: {result}")
    return result + " from consumer"


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
    )
    run = flyte.with_runcontext(
        log_level=logging.DEBUG,
        copy_style="none",
        version=flyte.__version__,
        # ).run(use_library, data="hello world", n=10)
    ).run(flyte_entities.library_parent_task, data="hello world", n=5)
    print(run.name)
    print(run.url)
    run.wait()
    # print(run.outputs())
