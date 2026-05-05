"""NotebookTask with multiple outputs."""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="multiple_outputs_example",
    image=flyte.Image.from_debian_base(name="multiple-outputs-example").clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    ),
)

text_analysis = NotebookTask(
    name="text_analysis",
    notebook_path="notebooks/multiple_outputs.ipynb",
    task_environment=env,
    inputs={"text": str, "n": int},
    outputs={"repeated": str, "word_count": int, "char_count": int},
)


@env.task
def multi_output_workflow(text: str = "hello world", n: int = 3) -> tuple[str, int, int]:
    repeated, word_count, char_count = text_analysis(text=text, n=n)
    return repeated, word_count, char_count


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", copy_style="all").run(multi_output_workflow, text="flyte is great", n=2)
    print(f"Run URL: {run.url}")
    print(f"Outputs: {run.outputs()}")
