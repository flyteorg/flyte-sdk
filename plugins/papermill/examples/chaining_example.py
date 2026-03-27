"""Chaining NotebookTasks in a workflow — output of one feeds into another."""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="chaining_example",
    image=flyte.Image.from_debian_base(name="chaining-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

step1 = NotebookTask(
    name="step1_add",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
)

step2 = NotebookTask(
    name="step2_add",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
)


@env.task
def chained_workflow(a: int = 1, b: float = 2.0, c: float = 3.0) -> float:
    # step1: a + b
    intermediate = step1(x=a, y=b)
    # step2: intermediate + c
    final = step2(x=int(intermediate), y=c)
    return final


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(chained_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
