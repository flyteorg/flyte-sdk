"""Mix regular Flyte tasks and NotebookTasks in the same workflow."""

from pathlib import Path

import flyte
from flyte._image import PythonWheels
from tasks import add, double
from tasks import env as tasks_env

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="mixed_workflow_example",
    image=flyte.Image.from_debian_base(name="mixed-workflow-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
    depends_on=[tasks_env],
)

notebook_add = NotebookTask(
    name="notebook_add",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
)


@env.task
def mixed_workflow(n: int = 5) -> float:
    # Regular task: double the input
    doubled = double(n=n)

    # Notebook task: add doubled + 100.0
    nb_result = notebook_add(x=doubled, y=100.0)

    # Regular task: add nb_result + 0.5
    final = add(a=nb_result, b=0.5)

    return final


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(mixed_workflow, n=7)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
