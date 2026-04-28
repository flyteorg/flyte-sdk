"""Inline NotebookTask — defined inside a workflow function, not at module level.

The resolver serializes the notebook path and type schemas directly into the
task spec at registration time, so no module import is needed at execution
time. This lets you define a NotebookTask wherever it makes sense rather than
being forced to assign it to a module-level variable.

NotebookTask behaves like a synchronous task.  Call it with ``nb(...)`` from a
sync task or ``await nb.aio(...)`` from an async task.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

env = flyte.TaskEnvironment(
    name="inline_example",
    image=flyte.Image.from_debian_base(name="inline-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)


@env.task
def sync_inline_workflow(x: int = 3, y: float = 1.5) -> int:
    from flyteplugins.papermill import NotebookTask

    nb = NotebookTask(
        name="add_numbers_sync_inline",
        notebook_path="notebooks/basic_math.ipynb",
        task_environment=env,
        inputs={"x": int, "y": float},
        outputs={"result": int},
    )
    return nb(x=x, y=y)


@env.task
async def async_inline_workflow(x: int = 3, y: float = 1.5) -> int:
    from flyteplugins.papermill import NotebookTask

    nb = NotebookTask(
        name="add_numbers_async_inline",
        notebook_path="notebooks/basic_math.ipynb",
        task_environment=env,
        inputs={"x": int, "y": float},
        outputs={"result": int},
    )
    # Use .aio() when calling a NotebookTask from an async task function
    return await nb.aio(x=x, y=y)


if __name__ == "__main__":
    flyte.init_from_config()
    run_sync = flyte.with_runcontext(mode="remote", copy_style="all").run(sync_inline_workflow)
    print(f"Run URL: {run_sync.url}")
    print(f"Outputs: {run_sync.outputs()}")

    run_async = flyte.with_runcontext(mode="remote", copy_style="all").run(async_inline_workflow)
    print(f"Run URL: {run_async.url}")
    print(f"Outputs: {run_async.outputs()}")
