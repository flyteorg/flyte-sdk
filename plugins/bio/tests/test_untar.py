"""End-to-end check for the untar wrapper.

Extracts ``kraken2.tar.gz`` and asserts each emitted file's MD5 matches
the expected values.

This module runs a single ``@env.task``: it fetches its fixture, calls
``untar``, and asserts the extracted files' MD5s.

Run it::

    uv run --project plugins/bio pytest plugins/bio/tests/test_untar.py
"""

import flyte
import pytest
from _utils import FileT, assert_md5_files, fixture_file, init_local_flyte, run_local_task

from flyteplugins.bio.untar import env as untar_env
from flyteplugins.bio.untar import untar

env = flyte.TaskEnvironment(name="untar_tests", depends_on=[untar_env])


@env.task
async def untar_case() -> None:
    archive = await fixture_file("genomics/sarscov2/genome/db/kraken2.tar.gz")
    files: list[FileT] = await untar(archive=archive)
    await assert_md5_files(
        "untar kraken2.tar.gz",
        files,
        {
            "hash.k2d": "8b8598468f54a7087c203ad0190555d9",
            "opts.k2d": "a033d00cf6759407010b21700938f543",
            "taxo.k2d": "094d5891cdccf2f1468088855c214b2c",
        },
    )


@pytest.fixture(scope="module", autouse=True)
def _init_flyte() -> None:
    init_local_flyte()


@pytest.mark.asyncio
async def test_untar() -> None:
    await run_local_task(untar_case)
