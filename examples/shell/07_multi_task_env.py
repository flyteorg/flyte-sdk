"""Multiple shell tasks sharing one TaskEnvironment.

When several ``shell.create()`` tasks share the same container image, group
them under a single :class:`flyte.TaskEnvironment` via ``from_task(...)``.
Pipelines then ``depends_on=[that_env]`` once and gain access to every task.

Here we wrap three trivial "text utilities" under a shared debian-slim image::

    upper  — uppercase a file
    lower  — lowercase a file
    rev    — reverse each line

Run locally::

    uv run python 07_multi_task_env.py
"""

import sys
import tempfile

import flyte
from flyte.extras import shell
from flyte.io import File

BASE = "debian:12-slim"


upper = shell.create(
    name="upper",
    image=BASE,
    inputs={"src": File},
    outputs={"out": File},
    script=r"tr '[:lower:]' '[:upper:]' < {inputs.src} > {outputs.out}",
)


lower = shell.create(
    name="lower",
    image=BASE,
    inputs={"src": File},
    outputs={"out": File},
    script=r"tr '[:upper:]' '[:lower:]' < {inputs.src} > {outputs.out}",
)


rev = shell.create(
    name="rev",
    image=BASE,
    inputs={"src": File},
    outputs={"out": File},
    script=r"rev < {inputs.src} > {outputs.out}",
)


# All three share debian:12-slim — group under one env. `from_task` enforces
# matching images and registers all tasks together.
text_utils_env = flyte.TaskEnvironment.from_task(
    "text_utils",
    upper.as_task(),
    lower.as_task(),
    rev.as_task(),
)


env = flyte.TaskEnvironment(name="shell_multi_task", depends_on=[text_utils_env])


@env.task
async def chained(src: File) -> tuple[File, File, File]:
    u = await upper(src=src)
    l_low = await lower(src=u)  # uppercased then lowercased -> back to original
    r = await rev(src=src)
    return u, l_low, r


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello World\nFlyte Shell\n")
        path = f.name

    run = flyte.with_runcontext(mode=mode).run(chained, File.from_local_sync(path))
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
