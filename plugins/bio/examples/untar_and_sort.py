"""Use ``untar`` and ``bedtools_sort`` together in one Flyte task.

Pass a tar archive containing a BED file, or let the example create a tiny
sample archive locally. The task extracts the archive with ``untar`` and then
sorts the BED with ``bedtools_sort``.

Run locally::

    uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py
    uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py /path/to/archive.tar.gz

Run remotely::

    uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py remote
    uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py remote /path/to/archive.tar.gz
"""

from __future__ import annotations

import asyncio
import io
import sys
import tarfile
from pathlib import Path
from typing import Literal, cast

import flyte
from flyte.io import File
from flyte.remote import Run

from flyteplugins.bio import env as bio_env
from flyteplugins.bio.bedtools import bedtools_sort
from flyteplugins.bio.untar import untar

Mode = Literal["local", "remote"]
_SAMPLE_BED = """\
chr2\t30\t40\tfeature_c
chr1\t20\t25\tfeature_b
chr1\t10\t15\tfeature_a
"""

env = flyte.TaskEnvironment(
    name="untar_and_sort_example",
    image=flyte.Image.from_debian_base(flyte_version="2.4.0"),
    depends_on=[bio_env],
)


def _usage() -> str:
    return (
        "Usage:\n"
        "  uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py\n"
        "  uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py /path/to/archive.tar.gz\n"
        "  uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py remote\n"
        "  uv run --project plugins/bio python plugins/bio/examples/untar_and_sort.py remote /path/to/archive.tar.gz"
    )


def _parse_args(argv: list[str]) -> tuple[Mode, Path | None]:
    args = argv[1:]
    mode: Mode = "local"
    if args and args[0] == "remote":
        mode = "remote"
        args = args[1:]
    if len(args) > 1:
        raise SystemExit(_usage())
    archive = Path(args[0]).expanduser() if args else None
    return mode, archive


async def _init_for_mode(mode: Mode) -> None:
    if mode == "local":
        await flyte.init.aio()
    else:
        await flyte.init_from_config.aio()


@env.task(cache="auto")
async def make_sample_archive() -> File:
    """Build a tiny tar.gz archive containing an unsorted BED file."""
    archive = File.new_remote(file_name="sample-bed.tar.gz")
    data = _SAMPLE_BED.encode("utf-8")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="regions/sample.bed")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    async with archive.open("wb") as fh:
        await fh.write(buffer.getvalue())
    return archive


@env.task
async def untar_and_sort_example(archive: File) -> File:
    """Extract a BED archive and sort the extracted BED file."""
    extracted = await untar(archive=archive)
    bed = next((f for f in extracted if f.name.endswith(".bed")), None)
    if bed is None:
        raise ValueError("Archive did not contain a .bed file")
    return await bedtools_sort(i=bed)


@env.task
async def untar_and_sort_sample() -> File:
    """Create a sample archive, extract it, and sort the BED file."""
    archive = await make_sample_archive()
    return await untar_and_sort_example(archive=archive)


async def main() -> None:
    mode, archive_path = _parse_args(sys.argv)
    await _init_for_mode(mode)

    runner = flyte.with_runcontext(mode=mode)
    if archive_path is None:
        run = cast(Run, await runner.run.aio(untar_and_sort_sample))
    else:
        archive = cast(File, await File.from_local(str(archive_path)))
        run = cast(Run, await runner.run.aio(untar_and_sort_example, archive=archive))

    await run.wait.aio()

    print(run.url if mode == "remote" else run)


if __name__ == "__main__":
    asyncio.run(main())
