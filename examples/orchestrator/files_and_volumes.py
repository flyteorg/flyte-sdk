"""
Files, directories and volumes through an orchestrator
======================================================

A ``flyte.orchestrator.task`` passes files, directories and volumes between
the tasks it calls, but never opens them itself: it runs in a sandbox with no
storage access. The tasks it calls open them as usual.

- A ``File`` or ``Dir`` is a named tuple of its ``path``, ``name``,
  ``format`` and ``hash`` in the orchestrator, which can read those fields.
- A ``Volume`` (a Pydantic model, as are dataclasses and dataframes) is
  opaque there: the orchestrator passes it on, unchanged, but cannot read it.

Volumes need ``flyteplugins-union`` locally and the Union mount broker on the
cluster (the ``volumes.union.ai`` CSI driver), which ``allow_volumes()``
binds to. Nothing has to be deployed first::

    python files_and_volumes.py
"""

import asyncio
import uuid
from pathlib import Path

from flyteplugins.union.io import ROVolume, Volume, allow_volumes

import flyte
import flyte.orchestrator
from flyte.io import Dir, File

image = (
    flyte.Image.from_debian_base(install_flyte=False, name="orchestrator-files-and-volumes")
    .with_pip_packages("flyteplugins-union>=0.20.2")
    .with_local_v2()  # last, so the local SDK wins over the pip layer's flyte dependency
)

env = flyte.TaskEnvironment(
    name="orchestrator-files-and-volumes",
    image=image,
    pod_template=allow_volumes(),
    # The volume mount client and its staging tmpfs count against the pod's memory.
    resources=flyte.Resources(cpu="1", memory="4Gi"),
)


@env.task
async def write_file(text: str) -> File:
    f = File.new_remote(file_name="greeting.txt")
    async with f.open("wb") as fh:
        await fh.write(text.encode("utf-8"))
    return f


@env.task
async def read_file(f: File) -> str:
    async with f.open("rb") as fh:
        return bytes(await fh.read()).decode("utf-8")


@env.task
async def write_dir(count: int) -> Dir:
    root = Path("/tmp/shards")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (root / f"shard-{i}.txt").write_text(f"shard {i}\n")
    return await Dir.from_local(str(root))


@env.task
async def read_dir(d: Dir) -> list[str]:
    local = Path(await d.download())
    return sorted(f"{p.name}: {p.read_text().strip()}" for p in local.iterdir())


@env.task
async def init_volume(name: str, text: str) -> ROVolume:
    vol = Volume.new(name=name)
    root = await vol.mount()
    (root / "notes.txt").write_text(text + "\n")
    return await vol.finalize(message="initial write")


@env.task
async def append_to_volume(vol: ROVolume, line: str) -> ROVolume:
    rw = await vol.fork(name=f"{vol.name}-appended")
    root = await rw.mount()
    with open(root / "notes.txt", "a") as fh:  # noqa: ASYNC230
        fh.write(line + "\n")
    return await rw.finalize(message=f"append {line!r}")


@env.task
async def read_volume(vol: ROVolume) -> str:
    root = await vol.mount()
    return (root / "notes.txt").read_text()


@flyte.orchestrator.task
async def pipeline(text: str, shards: int, volume_name: str) -> dict[str, str]:
    f, d, vol = await asyncio.gather(
        write_file(text),
        write_dir(shards),
        init_volume(volume_name, text),
    )
    # The orchestrator reads a File's or Dir's fields, but passes the volume on unopened.
    appended = await append_to_volume(vol, "appended by a task the orchestrator called")
    contents, listing, notes = await asyncio.gather(read_file(f), read_dir(d), read_volume(appended))
    return {
        "file": f.name + ": " + contents,
        "dir": d.name + ": " + ", ".join(listing),
        "volume": notes.strip().replace("\n", " | "),
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline, text="hello from the orchestrator", shards=3, volume_name=f"orch-{uuid.uuid4().hex[:8]}")
    print(run.url)
    run.wait()
    for key, value in run.outputs()[0].items():
        print(f"{key}: {value}")
