"""End-to-end checks for the bedtools shell wrappers.

Each scenario runs one bedtools wrapper with fixed fixture data and
checks the emitted output against an expected MD5.

Every scenario is a single ``@env.task``:
it fetches its fixtures, calls the matching ``bedtools_<sub>`` wrapper,
and asserts the output MD5 against the upstream snapshot.

Run it::

    uv run --project plugins/bio pytest plugins/bio/tests/test_bedtools.py
"""

import asyncio

import flyte
import pytest
from _utils import FileT, assert_md5, fixture_file, init_local_flyte, run_local_task

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
async def bedtools_intersect_case() -> None:
    a = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    b = await fixture_file("genomics/sarscov2/genome/bed/test2.bed")
    out: FileT = await bedtools_intersect(a=a, b=[b])
    await assert_md5("bedtools intersect", out, "afcbf01c2f2013aad71dbe8e34f2c15c")


@env.task
async def bedtools_sort_case() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    out: FileT = await bedtools_sort(i=i)
    await assert_md5("bedtools sort", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def bedtools_sort_with_genome_case() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    g = await fixture_file("genomics/sarscov2/genome/genome.fasta.fai")
    out: FileT = await bedtools_sort(i=i, g=g)
    await assert_md5("bedtools sort -g", out, "fe4053cf4de3aebbdfc3be2efb125a74")


@env.task
async def bedtools_merge_case() -> None:
    i = await fixture_file("genomics/sarscov2/genome/bed/test.bed")
    out: FileT = await bedtools_merge(i=i)
    await assert_md5("bedtools merge", out, "0cf6ed2b6f470cd44a247da74ca4fe4e")


@env.task
async def bedtools_suite() -> None:
    """Run every bedtools scenario as child tasks of a single run."""
    await asyncio.gather(
        bedtools_intersect_case(),
        bedtools_sort_case(),
        bedtools_sort_with_genome_case(),
        bedtools_merge_case(),
    )


@pytest.fixture(scope="module", autouse=True)
def _init_flyte() -> None:
    init_local_flyte()


@pytest.mark.asyncio
async def test_bedtools_intersect() -> None:
    await run_local_task(bedtools_intersect_case)


@pytest.mark.asyncio
async def test_bedtools_sort() -> None:
    await run_local_task(bedtools_sort_case)


@pytest.mark.asyncio
async def test_bedtools_sort_with_genome() -> None:
    await run_local_task(bedtools_sort_with_genome_case)


@pytest.mark.asyncio
async def test_bedtools_merge() -> None:
    await run_local_task(bedtools_merge_case)


@pytest.mark.asyncio
async def test_bedtools_suite() -> None:
    await run_local_task(bedtools_suite)
