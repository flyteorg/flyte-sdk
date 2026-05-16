"""Stream and scalar outputs — Stdout, Stderr, and primitive outputs.

Three patterns for "the tool's result is a value, not a file":

1. ``Stdout()`` — capture the script's stdout as a :class:`File` (default)
   or parse it as a primitive (``int``, ``float``, ``str``, ``bool``).
2. ``Stderr()`` — symmetric, but for stderr.
3. Bare primitive outputs (``int``, ``float``, ``str``, ``bool``) — the
   script writes a value to ``{outputs.<name>}``; the wrapper reads + casts.

Run locally::

    uv run python 03_streams.py
"""

import sys
import tempfile

import flyte
from flyte.extras import shell
from flyte.extras.shell import Stderr, Stdout
from flyte.io import File

# 1) Stdout as parsed int — `wc -l` writes "  42 filename" so we extract just the count.
line_count = shell.create(
    name="wc_lines",
    image="debian:12-slim",
    inputs={"src": File},
    outputs={"n": Stdout(type=int)},
    script=r"""
        wc -l < {inputs.src}
    """,
    debug=True,
)


# 2) Stdout as File (raw text) + Stderr as str (status messages from sort).
sort_with_diagnostics = shell.create(
    name="sort_with_diagnostics",
    image="debian:12-slim",
    inputs={"src": File},
    outputs={
        "sorted": Stdout(),  # the sorted file
        "diagnostics": Stderr(type=str),  # whatever sort wrote to stderr
    },
    script=r"""
        echo "sorting $(wc -l < {inputs.src}) lines" >&2
        sort {inputs.src}
    """,
    debug=True,
)


# 3) Bare primitive outputs — write to the canonical output path, parse on return.
divide = shell.create(
    name="divide",
    image="debian:12-slim",
    inputs={"a": int, "b": int},
    outputs={
        "quotient": float,
        "remainder": int,  # default path == output name
    },
    script=r"""
        # awk handles float division; remainder is %.
        awk "BEGIN {{print {inputs.a} / {inputs.b}}}" > {outputs.quotient}
        echo $(( {inputs.a} % {inputs.b} )) > {outputs.remainder}
    """,
    debug=True,
)


env = flyte.TaskEnvironment(
    name="shell_streams",
    depends_on=[line_count.env, sort_with_diagnostics.env, divide.env],
)


@env.task
async def stream_demo(src: File) -> tuple[int, File, str, float, int]:
    n = await line_count(src=src)
    sorted_file, diag = await sort_with_diagnostics(src=src)
    q, r = await divide(a=37, b=4)
    return n, sorted_file, diag, q, r


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Unsorted input; sort + wc both have something to do.
        f.write("charlie\nalpha\nbravo\ndelta\necho\nfoxtrot\n")
        path = f.name

    run = flyte.with_runcontext(mode=mode).run(stream_demo, File.from_local_sync(path))
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
