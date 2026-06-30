"""NotebookTask with File, Dir, and DataFrame types.

Demonstrates how complex Flyte types are passed to notebooks as serialized
path/URI strings, and reconstructed inside the notebook using load_file(),
load_dir(), and load_dataframe().
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import DataFrame, Dir, File

from flyteplugins.papermill import NotebookTask

env = flyte.TaskEnvironment(
    name="complex_types_example",
    image=flyte.Image.from_debian_base(name="complex-types-example")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-papermill",
        ),
    )
    .with_pip_packages("pandas", "pyarrow"),
)

process_file = NotebookTask(
    name="process_file",
    notebook_path="notebooks/process_file.ipynb",
    task_environment=env,
    inputs={"input_file": File},
    outputs={"line_count": int, "output_file": File},
)

process_dir = NotebookTask(
    name="process_dir",
    notebook_path="notebooks/process_dir.ipynb",
    task_environment=env,
    inputs={"input_dir": Dir},
    outputs={"file_count": int, "output_dir": Dir},
)

process_dataframe = NotebookTask(
    name="process_dataframe",
    notebook_path="notebooks/process_dataframe.ipynb",
    task_environment=env,
    inputs={"input_df": DataFrame},
    outputs={"row_count": int, "filtered_df": DataFrame},
)


@env.task
def prepare_file() -> File:
    """Create a sample file for the notebook to process."""
    with open("/tmp/sample.txt", "w") as fh:
        fh.write("hello\nworld\nfrom flyte\n")

    return File.from_local_sync("/tmp/sample.txt")


@env.task
def prepare_dir() -> Dir:
    """Create a sample directory with files for the notebook to process."""
    import os

    os.makedirs("/tmp/sample_dir", exist_ok=True)
    for name in ["a.txt", "b.txt", "c.txt"]:
        with open(f"/tmp/sample_dir/{name}", "w") as fh:
            fh.write(f"contents of {name}\n")

    return Dir.from_local_sync("/tmp/sample_dir/")


@env.task
def prepare_dataframe() -> DataFrame:
    """Create a sample DataFrame for the notebook to process."""
    import pandas as pd

    df = pd.DataFrame({"name": ["alice", "bob", "carol"], "score": [90, 75, 88]})
    return DataFrame.wrap_df(df)


@env.task
def file_workflow() -> tuple[int, File]:
    f = prepare_file()
    line_count, output_file = process_file(input_file=f)
    return line_count, output_file


@env.task
def dir_workflow() -> tuple[int, Dir]:
    d = prepare_dir()
    file_count, output_dir = process_dir(input_dir=d)
    return file_count, output_dir


@env.task
def dataframe_workflow() -> tuple[int, DataFrame]:
    df = prepare_dataframe()
    row_count, filtered_df = process_dataframe(input_df=df)
    return row_count, filtered_df


if __name__ == "__main__":
    flyte.init_from_config()
    run_file = flyte.with_runcontext(mode="remote", copy_style="all").run(file_workflow)
    print(f"Run URL: {run_file.url}")
    print(f"Outputs: {run_file.outputs()}")

    run_dir = flyte.with_runcontext(mode="remote", copy_style="all").run(dir_workflow)
    print(f"Run URL: {run_dir.url}")
    print(f"Outputs: {run_dir.outputs()}")

    run_df = flyte.with_runcontext(mode="remote", copy_style="all").run(dataframe_workflow)
    print(f"Run URL: {run_df.url}")
    print(f"Outputs: {run_df.outputs()}")
