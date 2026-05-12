"""flyteplugins.bio — curated bioinformatics CLI wrappers for Flyte.

See submodules:

- :mod:`flyteplugins.bio.bedtools` — bedtools genome arithmetic
  (intersect, sort, merge).

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

env = flyte.TaskEnvironment(
    name="bio",
    depends_on=[bedtools_env],
)

__all__ = ["bedtools_env", "env"]
