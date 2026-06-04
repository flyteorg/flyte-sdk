"""cat — file concatenation utilities.

Currently exposes:

- :data:`cat_fastq` — concatenate gzipped FASTQ shards into a single
  merged file, preserving input order.

The wrapper assumes inputs are already gzipped (.fastq.gz).
"""

from __future__ import annotations

import flyte
from flyte.extras import shell
from flyte.io import File

CAT_IMAGE = "community.wave.seqera.io/library/coreutils_grep_gzip_lbzip2_pruned:838ba80435a629f8"


cat_fastq = shell.create(
    name="cat_fastq",
    image=CAT_IMAGE,
    inputs={"reads": list[File]},
    outputs={"merged": File},
    script=r"""
        ls -1 -v {inputs.reads} | xargs cat > {outputs.merged}
    """,
)


env = flyte.TaskEnvironment.from_task("cat", cat_fastq.as_task())
