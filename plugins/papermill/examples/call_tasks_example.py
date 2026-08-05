"""NotebookTask that calls other Flyte tasks from within the notebook.

Locally, task calls run as regular function calls.
Remotely, they are submitted to Flyte and appear as separate nodes in the UI.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels
from tasks import env as tasks_env

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="call_tasks_example",
    image=flyte.Image.from_debian_base(name="call-tasks-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
    depends_on=[tasks_env],
)

notebook_with_tasks = NotebookTask(
    name="notebook_with_tasks",
    notebook_path="notebooks/call_tasks.ipynb",
    task_environment=env,
    inputs={"a": int, "b": int},
    outputs={"total": int},
)


@env.task
def task_calling_workflow(a: int = 10, b: int = 20) -> int:
    return notebook_with_tasks(a=a, b=b)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(task_calling_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
