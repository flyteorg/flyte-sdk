"""gunzip — decompress a single gzipped file.

Ports nf-core's ``gunzip`` module. Wraps ``gzip -cd`` (not the
``gunzip`` binary itself — see nf-core's note: ``gunzip`` writes the
output with the source file's group ownership, ``gzip -cd >`` doesn't).
"""

from __future__ import annotations

import flyte
from flyte.extras import shell
from flyte.io import File

# From nf-core/modules/gunzip/main.nf (Seqera wave coreutils image).
GUNZIP_IMAGE = "community.wave.seqera.io/library/coreutils_grep_gzip_lbzip2_pruned:838ba80435a629f8"



gunzip = shell.create(
    name="gunzip",
    image=GUNZIP_IMAGE,
    inputs={"archive": File},
    outputs={"out": File},
    script=r"""
        gzip -cd {inputs.archive} > {outputs.out}
    """,
)


env = flyte.TaskEnvironment.from_task(
    "gunzip",
    gunzip.as_task(),
)
