# flyteplugins-nextflow

Run [Nextflow](https://www.nextflow.io/) pipelines on Flyte.

The Nextflow head runs inside a Flyte task. The [nf-flyte](https://github.com/unionai/nf-flyte)
Nextflow plugin runs every Nextflow task as a child action of that task. A pipeline
therefore shows up in Flyte as one run with one action per Nextflow task:

- Each process runs in its own `container`, with its `cpus`, `memory`, `disk`,
  `accelerator` and `time` directives. The image doesn't need Flyte or the AWS CLI.
- Nextflow keeps its own retries (`errorStrategy`), caching (`-resume`) and file staging.
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
| `outdir`     | `--outdir`. A relative path is stored under the run's storage. The directory is returned as a `Dir`. |
| `work_dir`   | S3 work directory (default: under the run's storage). Use a fixed location with `resume=True` to resume across runs. |
| `config`     | Extra Nextflow config, as a file path or config text. |
| `resume`     | Pass `-resume`. |
| `extra_args` | Any other `nextflow run` arguments. |

If Nextflow fails, `run_nextflow` raises a `NextflowError` containing the end of the
Nextflow output and the end of `.nextflow.log`.

## Image

`nextflow_image()` builds the image for the task that runs the Nextflow head. It contains
a JRE, Nextflow (`nextflow_version`, default `25.10.6`) and the nf-flyte plugin. To use a
locally built plugin, pass its zip:

```python
from pathlib import Path

image = nextflow_image(nf_flyte=Path("~/nf-flyte/build/distributions/nf-flyte-0.1.0.zip").expanduser())
```

## Requirements

- Flyte v2 backed by S3. The work directory must be on S3.
- Every process container needs `bash`.
- Task pods need read/write access to the work directory.
