"""Resources, retries, timeout, cache, and pre-built images.

Real shell tasks need to declare what they consume and how to handle
failure. ``shell.create()`` forwards the standard task knobs to the
underlying ContainerTask:

- ``resources`` — CPU, memory, GPU. Bio tools have very specific memory
  profiles (a samtools sort over a large BAM wants many GB; bwa mem wants
  threads x a few GB per thread). Declaring this honestly is what gets
  you efficient scheduling.
- ``retries`` — auto-retry on failure. Useful for tools that occasionally
  fail on flaky inputs or transient infrastructure issues.
- ``timeout`` — kill the task after N seconds. Bounds runaway jobs.
- ``cache`` — Flyte's task cache. ``"auto"`` (default) hashes inputs and
  reuses previous outputs; ``"override"`` always reruns; ``"disable"``
  never caches.

On the image: ``shell.create()`` accepts either a pre-built URI string or
a :class:`flyte.Image` instance (layered: base + apt / pip / Dockerfile
layers). When you pass a ``flyte.Image``, the shell layer builds it
lazily on first call via :func:`flyte.build`, using the configured
builder (``cfg.image_builder``: ``"local"`` by default, ``"remote"`` when
opted in), and hands the resolved URI down to ContainerTask. The build is
memoised — subsequent calls reuse the URI.

Run locally::

    uv run python 10_resources_and_image.py
"""

import sys
import tempfile

import flyte
from flyte.extras import shell
from flyte.io import File

# A layered Image — base + extra apt package. shell.create builds this on
# first call via flyte.build (using the configured builder) and passes the
# resolved URI down to ContainerTask.
image_with_jq = flyte.Image.from_debian_base(install_flyte=False).with_apt_packages(
    "jq"
)


pretty_print_json = shell.create(
    name="pretty_print_json",
    image=image_with_jq,
    inputs={"src": File},
    outputs={"out": File},
    script=r"""
        jq . {inputs.src} > {outputs.out}
    """,
    # 2 cores, 4 GB — tune per-tool for real bio wrappers.
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    # Auto-retry up to 2 times on failure (flaky inputs, transient infra).
    retries=2,
    # Kill after 5 minutes. Bounds runaway invocations on bad input.
    timeout=300,
    # Cache hit on identical inputs reuses the output blob.
    cache="auto",
)


env = flyte.TaskEnvironment(
    name="shell_resources_image",
    depends_on=[pretty_print_json.env],
)


@env.task
async def reformat(src: File) -> File:
    return await pretty_print_json(src=src)


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"name":"flyte","feature":"shell","ok":true}')
        path = f.name

    run = flyte.with_runcontext(mode=mode).run(reformat, File.from_local_sync(path))
    print(run.url if mode == "remote" else run)
    print(f"Output: {run.outputs()}")
