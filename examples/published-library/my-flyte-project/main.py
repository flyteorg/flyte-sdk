import pathlib

from my_task_library import flyte_entities

import flyte

img = flyte.Image.from_debian_base().with_pip_packages("my-task-library")
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
    # You can run a task that uses library tasks normally. Note we still copy code bundle
    run = flyte.run(use_library, data="hello world", n=10)
    print(run.url)

    # You can also run tasks from the library directly, for efficiency do not copy the local code, just install the
    # package. Version is needed when code is not copied over.
    # Ideally pass the version = my_task_library.__version__
    run = flyte.with_runcontext(copy_style="none", version="0.4.0").run(
        flyte_entities.library_parent_task, data="hello world", n=5
    )
    print(run.url)
