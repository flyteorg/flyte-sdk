"""Demonstrates all papermill execution options available on NotebookTask."""

from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import File

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="papermill_options_example",
    image=flyte.Image.from_debian_base(name="papermill-options-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

# All available papermill options with their defaults shown inline:
all_options = NotebookTask(
    name="all_options",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
    # Jupyter kernel to use. None means use the kernel from the notebook metadata.
    kernel_name="python3",
    # Override the notebook language (rare — only needed if the kernel metadata
    # doesn't match the actual language).
    language=None,
    # Per-cell execution timeout in seconds. None means no timeout.
    execution_timeout=300,
    # Seconds to wait for the kernel to start before giving up.
    start_timeout=120,
    # Stream cell outputs (print, display) to the Flyte task log.
    log_output=True,
    # Show a tqdm-style progress bar during execution (visible in logs).
    progress_bar=True,
    # Hide input cells in the executed output notebook (only show outputs).
    report_mode=False,
    # Save the notebook after every cell execution (nbclient engine only).
    # In remote execution this is largely redundant: the Flyte
    # papermill plugin renders and uploads the partial output notebook even
    # when execution fails, so crash diagnostics don't depend on this.
    # Set to True only if you use a custom engine that relies on it.
    request_save_on_cell_execute=False,
    # Papermill engine name. Default uses nbclient. Custom engines can be
    # registered via the `papermill.engine` entry point.
    engine_name=None,
    # Extra keyword arguments forwarded directly to the papermill engine.
    # For example, `autosave_cell_every` (int) saves every N seconds.
    engine_kwargs={"autosave_cell_every": 30},
)

# Verbose debugging: log all cell output, long timeouts
verbose_debug = NotebookTask(
    name="verbose_debug",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
    log_output=True,
    start_timeout=300,
    execution_timeout=600,
    progress_bar=True,
)

# Fast execution: no logging, no progress bar, no cell-level saving
fast_execution = NotebookTask(
    name="fast_execution",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
    log_output=False,
    progress_bar=False,
    request_save_on_cell_execute=False,
)

# Clean report: papermill marks input cells with transient.remove_source=True,
# which our renderer respects — input cells are stripped from both the Report
# HTML and the uploaded .ipynb.
clean_report = NotebookTask(
    name="clean_report",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
    report_mode=True,
    output_notebooks=True,
)

# Return both notebooks as File outputs so they're visible in the pipeline
# and accessible to downstream tasks.
# output_notebooks=True adds two outputs:
#   output_notebook: the source .ipynb (no cell outputs)
#   output_notebook_executed: the .ipynb after papermill execution (with cell outputs)
# The HTML rendering still appears in the Report as usual.
with_notebook_output = NotebookTask(
    name="with_notebook_output",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
    output_notebooks=True,
)


@env.task
def options_workflow(x: int = 42, y: float = 0.5) -> tuple[float, float, float, float, float, File, File, File, File]:
    r1 = all_options(x=x, y=y)
    r2 = verbose_debug(x=x, y=y)
    r3 = fast_execution(x=x, y=y)
    r4, nb_clean, nb_executed_clean = clean_report(x=x, y=y)

    # result + source notebook + executed notebook (with cell outputs)
    r5, nb, nb_executed = with_notebook_output(x=x, y=y)
    return r1, r2, r3, r4, r5, nb_clean, nb_executed_clean, nb, nb_executed


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(options_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
