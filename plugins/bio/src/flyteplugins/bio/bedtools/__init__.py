"""bedtools — genome arithmetic CLI tools.

Wraps tools from the `bedtools suite <https://bedtools.readthedocs.io/>`_
shipped via the ``quay.io/biocontainers/bedtools`` image.

Currently exposed:

- :data:`bedtools_intersect` — report overlaps between two feature files.
- :data:`bedtools_sort` — sort BED/GFF/VCF features by chromosome and start.
- :data:`bedtools_merge` — combine overlapping or nearby features into one.

The module-level :data:`env` is a single :class:`flyte.TaskEnvironment`
containing every bedtools task. All commands share the same biocontainer
image, so they live in the same env. Pipelines depend on this one env to
gain access to every bedtools subcommand at once::

    from flyteplugins.bio.bedtools import (
        bedtools_intersect,
        bedtools_sort,
        bedtools_merge,
        env as bedtools_env,
    )

    env = flyte.TaskEnvironment(name="my_pipeline", depends_on=[bedtools_env])
"""

from __future__ import annotations

import flyte

from .intersect import bedtools_intersect
from .merge import bedtools_merge
from .sort import bedtools_sort

env = flyte.TaskEnvironment.from_task(
    "bedtools",
    bedtools_intersect.as_task(),
    bedtools_sort.as_task(),
    bedtools_merge.as_task(),
)

__all__ = [
    "bedtools_intersect",
    "bedtools_merge",
    "bedtools_sort",
    "env",
]
