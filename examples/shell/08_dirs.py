"""Dir input + Glob output — directory-shaped fan-out.

Many bio tools take a *directory* of files (think ``fastqc reads/``,
``multiqc results/``) and produce a *directory* of reports rather than one
file per output slot. Both halves are first-class:

- ``Dir`` as an input type — the whole directory is mounted at
  ``/var/inputs/<name>``; bash globs (``${name}/*``) iterate over its contents.
- ``Glob`` as an output type — the script writes files under
  ``/var/outputs/<name>/`` and the Python wrapper returns ``list[File]``.

Here we wrap a "per-file line counter": for each ``.txt`` in the input
directory, write a ``<basename>.summary`` file to the output directory
containing the line count. The serialized shell task still exposes that
output as a ``Dir`` remotely, but the surrounding Python task gets back a
``list[File]`` after the wrapper unpacks the globbed files.

Run locally::

    uv run python 08_dirs.py
"""

import sys
import tempfile
from pathlib import Path
from typing import Literal

import flyte
from flyte.extras import shell
from flyte.extras.shell import Glob
from flyte.io import Dir, File

summarize_dir = shell.create(
    name="summarize_dir",
    image="debian:12-slim",
    inputs={
        "src": Dir,  # mounted at /var/inputs/src
    },
    outputs={
        # The shell task writes files under /var/outputs/summaries/, then
        # the Python wrapper returns the matched files as list[File].
        "summaries": Glob("**/*"),
    },
    script=r"""
        for f in {inputs.src}/*; do
            base=$(basename "$f")
            echo "$base: $(wc -l < "$f") lines" > "{outputs.summaries}/${base}.summary"
        done
    """,
    cache="disable",
)


env = flyte.TaskEnvironment(name="shell_dirs", depends_on=[summarize_dir.env])


@env.task
async def summarize_files(src: Dir) -> list[File]:
    """Count lines in each file of ``src``, return one summary file per input."""
    return await summarize_dir(src=src)


if __name__ == "__main__":
    flyte.init_from_config()
    mode: Literal["local", "remote"] = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    # Build a small input directory locally.
    tmp = Path(tempfile.mkdtemp())
    for i, body in enumerate(["one\ntwo\nthree\n", "alpha\nbeta\n", "single\n"]):
        (tmp / f"file_{i}.txt").write_text(body)

    run = flyte.with_runcontext(mode=mode).run(summarize_files, Dir.from_local_sync(str(tmp)))
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
