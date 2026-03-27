"""Basic NotebookTask example — single input/output with a simple workflow."""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="basic_example",
    image=flyte.Image.from_debian_base(name="basic-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

add_numbers = NotebookTask(
    name="add_numbers",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
)


@env.task
def basic_workflow(x: int = 5, y: float = 3.14) -> float:
    return add_numbers(x=x, y=y)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(basic_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
