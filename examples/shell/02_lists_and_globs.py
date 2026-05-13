"""list[File] input + Glob output — fan-in and fan-out.

The tool ``split`` cuts a single concatenated file into N chunks; we feed it
a list of files (which the shell layer packs into a directory and renders as
a bash glob), and collect the resulting chunks back via :class:`Glob`.

This exercises the two trickiest type-translations in the shell layer:

- ``list[File]``: Materialized under ``${input_data_dir}/<name>/`` and
  expanded as ``${input_data_dir}/<name>/*`` inside the script.
- ``Glob`` outputs: The tool decides how many files to produce. The
  serialized shell task exposes that output as a ``Dir`` remotely, while
  the Python shell wrapper scans the output dir post-run and returns
  ``list[File]``.

That means a directly executed remote ``cat_and_split`` task may show a
directory-shaped output in the UI, but the surrounding ``split_concatenated``
Python task returns ``list[File]`` because it awaits the shell wrapper.

Run locally::

    uv run python 02_lists_and_globs.py
"""

import sys
import tempfile

import flyte
from flyte.extras import shell
from flyte.extras.shell import Glob
from flyte.io import File

split_task = shell.create(
    name="cat_and_split",
    image="debian:12-slim",
    inputs={
        "parts": list[File],  # concatenated sources
        "chunk_lines": int,  # split every N lines
    },
    outputs={
        # split produces files named chunk_aa, chunk_ab, chunk_ac, …
        # We don't know the count up front, so we use Glob.
        "chunks": Glob("chunk_*"),
    },
    script=r"""
        cat {inputs.parts} | split -l {inputs.chunk_lines} - {outputs.chunks}/chunk_
    """,
)


env = flyte.TaskEnvironment(name="shell_lists_and_globs", depends_on=[split_task.env])


@env.task
async def split_concatenated(parts: list[File], chunk_lines: int) -> list[File]:
    """Concatenate `parts` then split into ~`chunk_lines`-line chunks."""
    return await split_task(parts=parts, chunk_lines=chunk_lines)


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    parts: list[File] = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_p{i}.txt", delete=False) as f:
            f.write("\n".join(f"part-{i}/line-{j}" for j in range(10)))
            parts.append(File.from_local_sync(f.name))

    run = flyte.with_runcontext(mode=mode).run(split_concatenated, parts, 7)
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
