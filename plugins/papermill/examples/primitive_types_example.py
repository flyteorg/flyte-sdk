"""NotebookTask with bool, list, and dict parameter types.

Demonstrates passing bool, list, and dict inputs to a notebook via papermill.
These types are injected directly as JSON-serializable parameters — no
load_file() / load_dir() helpers needed inside the notebook.

Note on None: None can be passed as a value at call time (papermill injects it
as null), but NoneType is not a valid Flyte type annotation for inputs/outputs.
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="primitive_types_example",
    image=flyte.Image.from_debian_base(name="primitive-types-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

analyze_values = NotebookTask(
    name="analyze_values",
    notebook_path="notebooks/primitive_types.ipynb",
    task_environment=env,
    inputs={
        "enabled": bool,  # feature flag — controls whether computation runs
        "values": list[int] | None,  # list of numbers to aggregate
        "options": dict[str, int | str] | None,  # config: {"threshold": int, "label": str}
    },
    outputs={
        "count": int,  # number of values above threshold
        "total": float,  # sum of all values (0.0 if not enabled)
        "label": str,  # label from options dict
    },
)


@env.task
def analyze_workflow(
    enabled: bool = True,
    values: list | None = None,
    options: dict | None = None,
) -> tuple[int, float, str]:
    values = values or [1, 5, 10, 15, 20]
    options = options or {"threshold": 8, "label": "demo"}
    return analyze_values(enabled=enabled, values=values, options=options)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(analyze_workflow)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
