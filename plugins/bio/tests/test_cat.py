"""End-to-end check for the cat_fastq wrapper.

Concatenates ``test_1.fastq.gz`` + ``test_2.fastq.gz`` byte-for-byte
into a single merged ``.fastq.gz`` and asserts the MD5 matches the
expected value.

``test_cat`` is a single ``@env.task``: it fetches its fixtures, calls
``cat_fastq``, and asserts the output MD5. ``main`` runs it once and
waits (a no-op locally; on remote it surfaces the in-container assert
once the run reaches a terminal state).

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_cat.py            # local
    uv run --project plugins/bio python plugins/bio/tests/test_cat.py remote     # devbox / prod
"""

from __future__ import annotations

import asyncio

import flyte
from _utils import FileT, assert_md5, cli_mode, fixture_file, init_for_mode

from flyteplugins.bio.cat import cat_fastq
from flyteplugins.bio.cat import env as cat_env

env = flyte.TaskEnvironment(name="cat_tests", depends_on=[cat_env])


@env.task
async def test_cat() -> None:
    r1 = await fixture_file("genomics/sarscov2/illumina/fastq/test_1.fastq.gz")
    r2 = await fixture_file("genomics/sarscov2/illumina/fastq/test_2.fastq.gz")
    out: FileT = await cat_fastq(reads=[r1, r2])
    await assert_md5("cat_fastq single-end", out, "ee314a9bd568d06617171b0c85f508da")


async def main() -> None:
    mode = cli_mode()
    await init_for_mode(mode)
    runner = flyte.with_runcontext(mode=mode)
    run = await runner.run.aio(test_cat)
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
