from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="inline_example",
    image=flyte.Image.from_debian_base(name="inline-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)


nb = NotebookTask(
    name="add_numbers_sync_inline",
    notebook_path="notebooks/basic_math.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": int},
)


if __name__ == "__main__":
    flyte.init_from_config()
    run_sync = flyte.with_runcontext(mode="remote", copy_style="all").run(nb, x=3, y=1.5)
    print(f"Run URL: {run_sync.url}")
    print(f"Outputs: {run_sync.outputs()}")
