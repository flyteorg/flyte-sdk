"""untar — extract a tar archive."""

from __future__ import annotations

import flyte
from flyte.extras import shell
from flyte.extras.shell import Glob
from flyte.io import File

UNTAR_IMAGE = "community.wave.seqera.io/library/coreutils_grep_gzip_lbzip2_pruned:838ba80435a629f8"

untar = shell.create(
    name="untar",
    image=UNTAR_IMAGE,
    inputs={"archive": File},
    outputs={"files": Glob("**/*")},
    script=r"""
        if [[ $(tar -taf {inputs.archive} | grep -o -P "^.*?/" | uniq | wc -l) -eq 1 ]]; then
            tar -C {outputs.files} --strip-components 1 -xavf {inputs.archive}
        else
            tar -C {outputs.files} -xavf {inputs.archive}
        fi
    """,
)


env = flyte.TaskEnvironment.from_task("untar", untar.as_task())
