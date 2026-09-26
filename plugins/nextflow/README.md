# flyteplugins-nextflow

Run [Nextflow](https://www.nextflow.io/) pipelines on Flyte.

The Nextflow head runs inside a Flyte task. The [nf-flyte](https://github.com/unionai/nf-flyte)
Nextflow plugin runs every Nextflow task as a child action of that task. A pipeline
therefore shows up in Flyte as one run with one action per Nextflow task:

- Each process runs in its own `container`, with its `cpus`, `memory`, `disk`,
  `accelerator` and `time` directives. The image only needs `bash`: Flyte stages the task's files.
- Nextflow keeps its own retries (`errorStrategy`) and `-resume`; Flyte caches task results across runs.
- Aborting the Flyte run aborts the pipeline and its running tasks.

## Installation

```bash
pip install flyteplugins-nextflow
```

## Quick start

```python
import flyte
from flyte.io import Dir
from flyteplugins.nextflow import nextflow_image, run_nextflow

env = flyte.TaskEnvironment(name="nextflow", image=nextflow_image())


@env.task
async def demo() -> Dir:
    return await run_nextflow("nf-core/demo", revision="1.2.0", profile="test", outdir="results")
```

`run_nextflow` accepts:

| Argument     | Description |
|--------------|-------------|
| `pipeline`   | Repository (`nf-core/rnaseq`), URL or path to run. |
| `revision`   | Git tag, branch or commit (`-r`). |
| `profile`    | Config profiles (`-profile`). |
| `params`     | Pipeline parameters, passed as a params file. |
| `outdir`     | `--outdir`. A relative path goes under the task's raw data prefix. The directory is returned as a `Dir`. |
| `work_dir`   | Work directory. Default: `nextflow-work` under the task's raw data prefix. |
| `config`     | Extra Nextflow config, as a file path or config text. |
| `resume`     | Resume the previous run that used the same `work_dir`. |
| `report`     | Render Nextflow's execution report, timeline and DAG into the task's Flyte report. The task must be declared with `report=True`. |
| `extra_args` | Any other `nextflow run` arguments. |

If Nextflow fails, `run_nextflow` raises a `NextflowError` containing the end of the
Nextflow output and the end of `.nextflow.log`.

### Resume

Nextflow's resume state lives in the work directory, not in the task pod:
- the task cache (the cloud cache on object stores, `.nextflow` locally)
- a session ID derived from the work directory

This has two effects:
- **Retries resume automatically.** The default work directory is shared by all attempts of
  a task, so a retried task passes `-resume` and picks up where the failed attempt stopped.
- **Resuming across runs needs a fixed `work_dir`.** Pass the same `work_dir` to each run,
  and `resume=True` from the second run on:

  ```python
  await run_nextflow("nf-core/rnaseq", work_dir="s3://my-bucket/rnaseq-work", resume=True)
  ```

### Local runs

When the task runs locally (`flyte run --local`), Nextflow uses its own executors instead
of nf-flyte: `local` by default, or whatever `profile` or `config` selects, such as `docker`.
This needs `nextflow` on the `PATH`.

## Image

`nextflow_image()` builds the image for the task that runs the Nextflow head. It contains
a JRE, Nextflow (`nextflow_version`, default `25.10.6`) and the nf-flyte plugin. To use a
locally built plugin, pass its zip:

```python
from pathlib import Path

image = nextflow_image(nf_flyte=Path("~/nf-flyte/build/distributions/nf-flyte-0.1.0.zip").expanduser())
```

## Requirements

- Flyte v2. On a cluster, the work directory must be on an object store Flyte can read (S3, GCS or Azure); the default, under the task's raw data prefix, always is.
- Every process container needs `bash`.
