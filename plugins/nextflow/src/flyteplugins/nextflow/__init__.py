"""
Run Nextflow pipelines on Flyte.

The Nextflow head runs inside a Flyte task, and the `nf-flyte` Nextflow plugin turns every
Nextflow task into a child action of that task, so a whole pipeline is a single Flyte run.

```python
import flyte
from flyteplugins.nextflow import nextflow_image, run_nextflow

env = flyte.TaskEnvironment(name="nextflow", image=nextflow_image())

@env.task
async def demo() -> flyte.io.Dir:
    return await run_nextflow("nf-core/demo", revision="1.0.2", profile="test", outdir="results")
```
"""

__all__ = [
    "DEFAULT_NEXTFLOW_VERSION",
    "DEFAULT_NF_FLYTE_VERSION",
    "NextflowError",
    "nextflow_image",
    "run_nextflow",
]

from flyteplugins.nextflow._image import DEFAULT_NEXTFLOW_VERSION, DEFAULT_NF_FLYTE_VERSION, nextflow_image
from flyteplugins.nextflow._run import NextflowError, run_nextflow
