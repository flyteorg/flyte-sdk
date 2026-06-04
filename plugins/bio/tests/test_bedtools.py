"""End-to-end checks for the bedtools shell wrappers.

Each scenario runs one bedtools wrapper with fixed fixture data and
checks the emitted output against an expected MD5.

Every scenario is a single ``@env.task`` named ``test_bedtools_<sub>``:
it fetches its fixtures, calls the matching ``bedtools_<sub>`` wrapper,
and asserts the output MD5 against the upstream snapshot.

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_bedtools.py

Pass ``remote`` as the first argument to submit runs to a cluster
(devbox or production); otherwise execution stays in-process. Inputs
are fetched on first run and cached under ``tests/_fixtures/``.
"""

import asyncio

import flyte
from _utils import FileT, assert_md5, cli_mode, fixture_file, init_for_mode

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
    a = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    b = await fixture_file("genomics/sarscov2/genome/bed/test2.bed")
    out: FileT = await bedtools_intersect(a=a, b=[b])
    await assert_md5("bedtools intersect", out, "afcbf01c2f2013aad71dbe8e34f2c15c")


@env.task
async def test_bedtools_sort() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    out: FileT = await bedtools_sort(i=i)
    await assert_md5("bedtools sort", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def test_bedtools_sort_with_genome() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    g = await fixture_file("genomics/sarscov2/genome/genome.fasta.fai")
    out: FileT = await bedtools_sort(i=i, g=g)
    await assert_md5("bedtools sort -g", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def test_bedtools_merge() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    out: FileT = await bedtools_merge(i=i)
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
    mode = cli_mode()
    await init_for_mode(mode)
    runner = flyte.with_runcontext(mode=mode)
    run = await runner.run.aio(test_bedtools)
    # In remote mode run.aio() only submits; wait so the child tasks'
    # in-container asserts (pass or fail) are surfaced here. A no-op locally,
    # where the run already executed in-process.
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
