"""End-to-end checks for the bedtools shell wrappers.

Each scenario mirrors a corresponding test in nf-core's bedtools modules
(``modules/nf-core/bedtools/{intersect,sort,merge}/tests``): same
biocontainer, same input fixtures, same expected output MD5 from
upstream snapshots.

Every scenario is a single ``@env.task`` named ``test_bedtools_<sub>``:
it fetches its fixtures, calls the matching ``bedtools_<sub>`` wrapper,
and asserts the output MD5 against the upstream snapshot.

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_bedtools.py

``flyte.init_from_config`` reads ``~/.flyte/config.yaml``. Pass ``remote``
as the first argument to submit runs to a cluster (devbox or
production); otherwise execution stays in-process. Inputs are fetched
from nf-core/test-datasets on first run and cached under
``tests/_fixtures/``; subsequent runs are offline.
"""

import asyncio
from typing import cast

import flyte
from _utils import FileT, assert_md5, cli_mode, nf_core_file
from flyte.remote import Run

from flyteplugins.bio.bedtools import (
    bedtools_intersect,
    bedtools_merge,
    bedtools_sort,
)
from flyteplugins.bio.bedtools import (
    env as bedtools_env,
)

env = flyte.TaskEnvironment(name="bedtools_tests", depends_on=[bedtools_env])


@env.task
async def test_bedtools_intersect() -> None:
    # nf-core: bedtools/intersect "sarscov2 - bed - bed"
    a = await nf_core_file("genomics/sarscov2/genome/bed/test.bed")
    b = await nf_core_file("genomics/sarscov2/genome/bed/test2.bed")
    out = cast(FileT, await bedtools_intersect(a=a, b=[b]))
    await assert_md5("bedtools intersect", out, "afcbf01c2f2013aad71dbe8e34f2c15c")


@env.task
async def test_bedtools_sort() -> None:
    # nf-core: bedtools/sort "test_bedtools_sort"
    i = await nf_core_file("genomics/sarscov2/genome/bed/test.bed")
    out = cast(FileT, await bedtools_sort(i=i))
    await assert_md5("bedtools sort", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def test_bedtools_sort_with_genome() -> None:
    # nf-core: bedtools/sort "test_bedtools_sort_with_genome"
    i = await nf_core_file("genomics/sarscov2/genome/bed/test.bed")
    g = await nf_core_file("genomics/sarscov2/genome/genome.fasta.fai")
    out = cast(FileT, await bedtools_sort(i=i, g=g))
    await assert_md5("bedtools sort -g", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def test_bedtools_merge() -> None:
    # nf-core: bedtools/merge "test_bedtools_merge"
    i = await nf_core_file("genomics/sarscov2/genome/bed/test.bed")
    out = cast(FileT, await bedtools_merge(i=i))
    await assert_md5("bedtools merge", out, "0cf6ed2b6f470cd44a247da74ca4fe4e")


@env.task
async def test_bedtools() -> None:
    """Run every bedtools scenario as child tasks of a single run."""
    await asyncio.gather(
        test_bedtools_intersect(),
        test_bedtools_sort(),
        test_bedtools_sort_with_genome(),
        test_bedtools_merge(),
    )


async def main() -> None:
    await flyte.init_from_config.aio()
    runner = flyte.with_runcontext(mode=cli_mode())
    run = cast(Run, await runner.run.aio(test_bedtools))
    # In remote mode run.aio() only submits; wait so the child tasks'
    # in-container asserts (pass or fail) are surfaced here. A no-op locally,
    # where the run already executed in-process.
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
