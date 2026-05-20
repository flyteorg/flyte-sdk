# flyteplugins-bio

Curated bioinformatics CLI tool wrappers for Flyte.

> [!WARNING]
> This plugin is still experimental and is not ready for general use or
> publication yet. The APIs, module layout, and packaging details may still
> change.

A collection of typed Flyte tasks for popular bioinformatics tools — bedtools,
samtools, bcftools, GATK, Picard, BWA, STAR, salmon, kallisto, and friends —
each shipped via its official image. Built on `flyte.extras.shell`.

Each submodule exposes:

- One or more callable shell tasks (`bedtools_intersect`, `samtools_view`, ...)
- A module-level `env` (a `flyte.TaskEnvironment`) that pipelines add to their
  `depends_on`. Adding the env causes the deploy pipeline to register the task
  and pull/build the underlying biocontainer image.

The package root also exposes `flyteplugins.bio.env`, an aggregate
environment that depends on every tool-family env. That gives pipelines a
single dependency when they want broad access to the bio plugin suite.

## Install

```bash
pip install flyteplugins-bio
```

Depends only on `flyte`. The actual bio tools come from biocontainers pulled at
task execution time — nothing to install locally.

## Usage

```python
import flyte
from flyte.io import File
from flyteplugins.bio import env as bio_env
from flyteplugins.bio.bedtools import bedtools_intersect

env = flyte.TaskEnvironment(
    name="genomics_pipeline",
    depends_on=[bio_env],
)

@env.task
async def pipeline(annotation: File, peaks: list[File]) -> list[File]:
    return await bedtools_intersect(a=annotation, b=peaks, wa=True, f=0.5)
```

## Currently available

- `flyteplugins.bio.bedtools` — `bedtools_intersect`, `bedtools_sort`, `bedtools_merge`

More tools are added as needed. Contributions following the same pattern (one
file per tool family, sharing one biocontainer image, exposing a module-level
`env`) are welcome once the plugin is ready to stabilize. The package-level
`flyteplugins.bio.env` is intended to grow alongside them.
