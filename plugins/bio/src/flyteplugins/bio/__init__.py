"""flyteplugins.bio — curated bioinformatics CLI wrappers for Flyte.

See submodules:

- :mod:`flyteplugins.bio.bedtools` — bedtools genome arithmetic
  (intersect, sort, merge).
- :mod:`flyteplugins.bio.cat` — concatenate FASTQ files.
- :mod:`flyteplugins.bio.gunzip` — decompress gzipped files.
- :mod:`flyteplugins.bio.untar` — extract tar archives.

Each submodule exposes callable shell tasks and a module-level ``env``
(``TaskEnvironment``) to plug into a pipeline's ``depends_on``.

For convenience, this package also exposes a top-level :data:`env` that
aggregates every bio tool-family environment. Pipelines that want access
to all installed bio modules can depend on one env instead of listing
each tool family individually::

    import flyte
    from flyteplugins.bio import env as bio_env

    env = flyte.TaskEnvironment(
        name="genomics_pipeline",
        depends_on=[bio_env],
    )
"""

from __future__ import annotations

import flyte

from .bedtools import env as bedtools_env
from .cat import env as cat_env
from .gunzip import env as gunzip_env
from .untar import env as untar_env

env = flyte.TaskEnvironment(
    name="bio",
    depends_on=[bedtools_env, cat_env, gunzip_env, untar_env],
)

__all__ = ["bedtools_env", "cat_env", "env", "gunzip_env", "untar_env"]
