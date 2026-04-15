# flyteplugins-papermill

Run Jupyter notebooks as Flyte tasks using [papermill](https://papermill.readthedocs.io/).

## Installation

```bash
pip install flyteplugins-papermill
```

## Quick start

```python
from flyteplugins.papermill import NotebookTask
import flyte

env = flyte.TaskEnvironment(name="my-env", image=flyte.Image.from_debian_base(name="my-env"))

notebook = NotebookTask(
    name="my_notebook",
    notebook_path="notebooks/analysis.ipynb",
    task_environment=env,
    inputs={"x": int, "y": float},
    outputs={"result": float},
)

@env.task
def workflow(x: int = 5, y: float = 3.14) -> float:
    return notebook(x=x, y=y)
```

### Notebook setup

Your notebook needs two tagged cells:

**`parameters` tag** — default input values (papermill injects actual values after this cell):

```python
x = 0
y = 0.0
```

**`outputs` tag** — records outputs using `record_outputs`:

```python
from flyteplugins.papermill import record_outputs
record_outputs(result=x + y)
```

To tag a cell in JupyterLab: View -> Right Sidebar -> Property Inspector -> Add Tag.

## Complex types: File, Dir, DataFrame

Flyte's `File`, `Dir`, and `DataFrame` types are serialized to path/URI strings when passed to papermill. Use the provided helpers inside the notebook to reconstruct them:

```python
from flyteplugins.papermill import load_file, load_dir, load_dataframe

# input_file, input_dir, input_df injected as strings by papermill
f = load_file(input_file)       # → flyte.io.File
d = load_dir(input_dir)         # → flyte.io.Dir
df = load_dataframe(input_df)   # → flyte.io.DataFrame
```

Use `await` for async I/O inside notebook cells (Jupyter supports top-level `await`):

```python
import pandas as pd

pdf = await df.open(pd.DataFrame).all()
output_df = await DataFrame.from_local(pdf)
```

Sync helpers (`open_sync`, etc.) work for `File` since they use fsspec directly, but DataFrame and Dir async methods require `await`.

## Calling Flyte tasks from notebooks

Tasks can be called from within notebooks. When running remotely, calls are submitted to Flyte and appear as separate tasks in the UI:

```python
from my_module import my_task

result = await my_task(input_value=42)
```

The Flyte runtime context is automatically injected into the notebook kernel — task calls route through the correct controller without any extra setup.

## Notebook outputs as pipeline artifacts (`output_notebooks`)

By default the executed notebook is rendered as an HTML report in the Flyte Report but is not visible as a pipeline artifact. Set `output_notebooks=True` to upload both notebooks to remote storage and return them as typed `File` outputs:

```python
notebook = NotebookTask(
    name="my_notebook",
    notebook_path="notebooks/analysis.ipynb",
    task_environment=env,
    inputs={"x": int},
    outputs={"result": float},
    output_notebooks=True,
)

@env.task
def workflow(x: int = 5) -> tuple[float, File, File]:
    result, source_nb, executed_nb = notebook(x=x)
    return result, source_nb, executed_nb
```

Two extra outputs are added automatically:

- `output_notebook` — the source `.ipynb` (no cell outputs)
- `output_notebook_executed` — the executed `.ipynb` (with cell outputs)

The HTML report still appears in the Report as usual.

## Clean reports (`report_mode`)

Setting `report_mode=True` tells papermill to mark input cells with `source_hidden` metadata. The plugin strips those cells from both the Report HTML and the uploaded `.ipynb` files, so only cell outputs are visible:

```python
notebook = NotebookTask(
    ...
    report_mode=True,
    output_notebooks=True,
)
```

## Execution report on failure

The report is rendered even when the notebook fails. Papermill writes the output notebook cell-by-cell, so the partial notebook is available on disk after a failure. The HTML report is flushed to the Flyte Report before the error is re-raised, giving full visibility into which cell failed and what output it produced.

## Spark notebooks

Use `plugin_config=Spark(...)` to run a notebook inside a Spark driver pod on Kubernetes:

```python
from flyteplugins.papermill import NotebookTask
from flyteplugins.spark import Spark

spark_nb = NotebookTask(
    name="spark_analysis",
    notebook_path="notebooks/spark_analysis.ipynb",
    task_environment=env,
    plugin_config=Spark(
        spark_conf={
            "spark.executor.instances": "2",
            "spark.executor.memory": "2g",
        }
    ),
    inputs={"data": list},
    outputs={"total": int, "count": int},
)
```

Inside the notebook, create the `SparkSession` directly:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FlyteSpark").getOrCreate()
```

> **Note:** `SparkContext.addPyFile()` is not called for notebook tasks. The notebook kernel runs in a subprocess that cannot share state with the parent task process, so in-process session setup is skipped. For K8s Spark this is not a limitation — executor pods use the same Docker image as the driver, so all packages are available on executors via the image. Dynamic code distribution via `addPyFile` is not supported.

## Local testing

Call a `NotebookTask` directly in Python (outside a Flyte workflow runner) for local testing:

```python
result = notebook_task(x=1, y=2.5)
```

This runs the notebook synchronously via papermill and returns the Python outputs. No report is rendered (requires a task context), no uploads happen, and no plugin hooks are called.

## Calling from async tasks

`NotebookTask` is a synchronous task (papermill blocks while the notebook runs). Call it with `nb(...)` from a sync task or `await nb.aio(...)` from an async task:

```python
# Sync task — call directly
@env.task
def workflow(x: int) -> float:
    return notebook(x=x)

# Async task — use .aio()
@env.task
async def workflow(x: int) -> float:
    return await notebook.aio(x=x)
```

## Inline definition

`NotebookTask` can be defined inside a task function rather than at module level. The resolver bakes the notebook path and type schemas into the task spec at registration time, so no module import is needed at execution time:

```python
@env.task
def workflow(x: int = 3, y: float = 1.5) -> int:
    from flyteplugins.papermill import NotebookTask

    nb = NotebookTask(
        name="add_numbers",
        notebook_path="notebooks/basic_math.ipynb",
        task_environment=env,
        inputs={"x": int, "y": float},
        outputs={"result": int},
    )
    return nb(x=x, y=y)
```

## NotebookTask reference

| Parameter                      | Default | Description                                                        |
| ------------------------------ | ------- | ------------------------------------------------------------------ |
| `name`                         | —       | Task name                                                          |
| `notebook_path`                | —       | Path to `.ipynb`, relative to the calling file or absolute         |
| `task_environment`             | —       | `TaskEnvironment` for registration and remote execution            |
| `inputs`                       | `None`  | `{name: type}` dict of notebook inputs                             |
| `outputs`                      | `None`  | `{name: type}` dict of notebook outputs                            |
| `plugin_config`                | `None`  | Plugin config (e.g. `Spark(...)`)                                  |
| `kernel_name`                  | `None`  | Jupyter kernel name; `None` uses the kernel from notebook metadata |
| `engine_name`                  | `None`  | Papermill engine; `None` uses the default `nbclient` engine        |
| `log_output`                   | `False` | Stream cell output to the task log                                 |
| `start_timeout`                | `60`    | Seconds to wait for kernel startup                                 |
| `execution_timeout`            | `None`  | Per-cell timeout in seconds; `None` means no timeout               |
| `report_mode`                  | `False` | Strip input cells from the Report HTML and uploaded `.ipynb`       |
| `request_save_on_cell_execute` | `True`  | Save notebook after every cell (nbclient engine only)              |
| `progress_bar`                 | `True`  | Show a tqdm-style progress bar during execution                    |
| `language`                     | `None`  | Override notebook language (rarely needed)                         |
| `engine_kwargs`                | `{}`    | Extra kwargs forwarded to the papermill engine                     |
| `output_notebooks`             | `False` | Upload source and executed `.ipynb` as `File` task outputs         |

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- [`basic_example.py`](examples/basic_example.py) — Single input/output
- [`multiple_outputs_example.py`](examples/multiple_outputs_example.py) — Multiple notebook outputs
- [`no_outputs_example.py`](examples/no_outputs_example.py) — Side-effect-only notebook
- [`complex_types_example.py`](examples/complex_types_example.py) — `File`, `Dir`, and `DataFrame` inputs/outputs
- [`call_tasks_example.py`](examples/call_tasks_example.py) — Calling Flyte tasks from within a notebook
- [`async_example.py`](examples/async_example.py) — Calling async tasks from within a notebook
- [`inline_example.py`](examples/inline_example.py) — Defining `NotebookTask` inline inside a task function
- [`chaining_example.py`](examples/chaining_example.py) — Chaining multiple notebooks
- [`mixed_workflow_example.py`](examples/mixed_workflow_example.py) — Mixing `NotebookTask` with regular tasks
- [`spark_example.py`](examples/spark_example.py) — Spark notebooks via `plugin_config`
- [`papermill_options_example.py`](examples/papermill_options_example.py) — All papermill execution options
