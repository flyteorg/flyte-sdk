"""NotebookTask calling async Flyte tasks from within the notebook.

Jupyter supports top-level `await`, so async tasks can be called directly
inside the notebook. The workflow itself is synchronous because papermill
execution is always synchronous.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="async_example",
    image=flyte.Image.from_debian_base(name="async-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)


@env.task
async def async_double(n: int) -> int:
    return n * 2


@env.task
async def async_add(a: float, b: float) -> float:
    return a + b


notebook = NotebookTask(
    name="notebook_async",
    notebook_path="notebooks/async_tasks.ipynb",
    task_environment=env,
    inputs={"a": float, "b": float},
    outputs={"result": float},
    log_output=True,
)


@env.task
def async_workflow(a: float = 20.0, b: float = 50.0) -> float:
    return notebook(a=a, b=b)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(async_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
