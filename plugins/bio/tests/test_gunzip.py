"""End-to-end check for the gunzip wrapper.

Mirrors nf-core's ``gunzip`` "Should run without failures" snapshot:
decompresses ``test_1.fastq.gz`` and asserts the output MD5 matches the
upstream snapshot value.

``test_gunzip`` is a single ``@env.task``: it fetches its fixture, calls
``gunzip``, and asserts the output MD5. ``main`` runs it once and waits.

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_gunzip.py            # local
    uv run --project plugins/bio python plugins/bio/tests/test_gunzip.py remote     # devbox / prod
"""

from __future__ import annotations

import asyncio
from typing import cast

import flyte
from _utils import FileT, assert_md5, cli_mode, nf_core_file
from flyte.remote import Run

from flyteplugins.bio.gunzip import env as gunzip_env
from flyteplugins.bio.gunzip import gunzip

env = flyte.TaskEnvironment(name="gunzip_tests", depends_on=[gunzip_env])


@env.task
async def test_gunzip() -> None:
    # nf-core: gunzip "Should run without failures"
    archive = await nf_core_file("genomics/sarscov2/illumina/fastq/test_1.fastq.gz")
    out = cast(FileT, await gunzip(archive=archive))
    await assert_md5("gunzip test_1.fastq.gz", out, "4161df271f9bfcd25d5845a1e220dbec")


async def main() -> None:
    await flyte.init_from_config.aio()
    runner = flyte.with_runcontext(mode=cli_mode())
    run = cast(Run, await runner.run.aio(test_gunzip))
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
