"""
Run the nf-core/demo pipeline (FastQC, seqtk trim, MultiQC) on Flyte.

Each pipeline process runs as a child action of `nf_core_demo`, in the process's own
biocontainers image, and the MultiQC report and other results come back as a `Dir`.

    flyte run examples/nf_core_demo.py nf_core_demo
"""

from flyteplugins.nextflow import nextflow_image, run_nextflow

import flyte
from flyte.io import Dir

env = flyte.TaskEnvironment(
    name="nf_core_demo",
    image=nextflow_image(),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


@env.task
async def nf_core_demo(revision: str = "1.2.0") -> Dir:
    # the `test` profile brings its own small samplesheet and resource limits
    return await run_nextflow("nf-core/demo", revision=revision, profile="test", outdir="results")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(nf_core_demo)
    print(run.url)
