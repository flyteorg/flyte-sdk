"""Notebook failure with partial report rendering.

Demonstrates that when a notebook fails mid-execution, the Flyte Report
is still populated with all cells that ran successfully before the error.

Papermill writes the output notebook cell-by-cell as execution progresses,
so the partial notebook is always available on disk. The plugin renders it to
HTML and flushes it to the Report before re-raising the exception — meaning
the report tab shows exactly which cell failed and what output the earlier
cells produced.

This is useful for debugging long-running notebooks: you can inspect the
partial results without re-running the whole notebook.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="failure_report_example",
    image=flyte.Image.from_debian_base(name="failure-report-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

# This notebook deliberately raises an error partway through.
# Flyte report will show the output from cells that ran before the failure.
failing_notebook = NotebookTask(
    name="failing_notebook",
    notebook_path="notebooks/partial_failure.ipynb",
    task_environment=env,
    inputs={"n": int},
    outputs={"result": int},
)


@env.task
def failure_workflow(n: int = 5) -> int:
    return failing_notebook(n=n)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(failure_workflow)
    print(f"Run URL: {run.url}")
