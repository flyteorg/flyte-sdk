"""End-to-end check for the cat_fastq wrapper.

Concatenates ``test_1.fastq.gz`` + ``test_2.fastq.gz`` byte-for-byte
into a single merged ``.fastq.gz`` and asserts the MD5 matches the
expected value.

This module runs a single ``@env.task``: it fetches its fixtures, calls
``cat_fastq``, and asserts the output MD5.

Run it::

    uv run --project plugins/bio pytest plugins/bio/tests/test_cat.py
"""

import flyte
import pytest
from _utils import FileT, assert_md5, fixture_file, init_local_flyte, run_local_task

from flyteplugins.bio.cat import cat_fastq
from flyteplugins.bio.cat import env as cat_env

env = flyte.TaskEnvironment(name="cat_tests", depends_on=[cat_env])


@env.task
async def cat_case() -> None:
    r1 = await fixture_file("genomics/sarscov2/illumina/fastq/test_1.fastq.gz")
    r2 = await fixture_file("genomics/sarscov2/illumina/fastq/test_2.fastq.gz")
    out: FileT = await cat_fastq(reads=[r1, r2])
    await assert_md5("cat_fastq single-end", out, "ee314a9bd568d06617171b0c85f508da")


@pytest.fixture(scope="module", autouse=True)
def _init_flyte() -> None:
    init_local_flyte()


@pytest.mark.asyncio
async def test_cat() -> None:
    await run_local_task(cat_case)
