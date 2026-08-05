"""NotebookTask with no outputs — useful for side effects like reports.

The executed notebook is automatically rendered as an HTML report
in the Flyte UI.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="no_outputs_example",
    image=flyte.Image.from_debian_base(name="no-outputs-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

printer = NotebookTask(
    name="printer",
    notebook_path="notebooks/no_outputs.ipynb",
    task_environment=env,
    inputs={"message": str, "count": int},
)


@env.task
def report_workflow(message: str = "hello", count: int = 5):
    printer(message=message, count=count)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(
        report_workflow, message="testing papermill", count=3
    )
    print(f"Run URL: {run.url}")
