"""Dir input + Dir output — directory-shaped data.

Many bio tools take a *directory* of files (think ``fastqc reads/``,
``multiqc results/``) and produce a *directory* of reports rather than one
file per output slot. Both halves are first-class:

- ``Dir`` as an input type — the whole directory is mounted at
  ``/var/inputs/<name>``; bash globs (``${name}/*``) iterate over its contents.
- ``Dir`` as an output type — the script creates a directory under
  ``/var/outputs`` and the wrapper returns it as a ``Dir`` handle.

Here we wrap a "per-file line counter": for each ``.txt`` in the input
directory, write a ``<basename>.summary`` file to the output directory
containing the line count. The pipeline gets back a single ``Dir`` referencing
all those summary files.

Run locally::

    uv run python 08_dirs.py
"""

import sys
import tempfile
from pathlib import Path

import flyte
from flyte.extras import shell
from flyte.io import Dir

summarize_dir = shell.create(
    name="summarize_dir",
    image="debian:12-slim",
    inputs={
        "src": Dir,  # mounted at /var/inputs/src
    },
    outputs={
        # /var/outputs/summaries is the canonical path the wrapper
        # pre-creates and CoPilot reads back as a ``Dir``.
        "summaries": Dir,
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
async def summarize_files(src: Dir) -> Dir:
    """Count lines in each file of ``src``, return a Dir of per-file summaries."""
    return await summarize_dir(src=src)


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    # Build a small input directory locally.
    tmp = Path(tempfile.mkdtemp())
    for i, body in enumerate(["one\ntwo\nthree\n", "alpha\nbeta\n", "single\n"]):
        (tmp / f"file_{i}.txt").write_text(body)

    run = flyte.with_runcontext(mode=mode).run(summarize_files, Dir.from_local_sync(str(tmp)))
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
