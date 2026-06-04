"""End-to-end check for the gunzip wrapper.

Decompresses ``test_1.fastq.gz`` and asserts the output MD5 matches the
expected value.

This module runs a single ``@env.task``: it fetches its fixture, calls
``gunzip``, and asserts the output MD5.

Run it::

    uv run --project plugins/bio pytest plugins/bio/tests/test_gunzip.py
"""

import flyte
import pytest
from _utils import FileT, assert_md5, fixture_file, init_local_flyte, run_local_task

from flyteplugins.bio.gunzip import env as gunzip_env
from flyteplugins.bio.gunzip import gunzip

env = flyte.TaskEnvironment(name="gunzip_tests", depends_on=[gunzip_env])


@env.task
async def gunzip_case() -> None:
    archive = await fixture_file("genomics/sarscov2/illumina/fastq/test_1.fastq.gz")
    out: FileT = await gunzip(archive=archive)
    await assert_md5("gunzip test_1.fastq.gz", out, "4161df271f9bfcd25d5845a1e220dbec")


@pytest.fixture(scope="module", autouse=True)
def _init_flyte() -> None:
    init_local_flyte()


@pytest.mark.asyncio
async def test_gunzip() -> None:
    await run_local_task(gunzip_case)
