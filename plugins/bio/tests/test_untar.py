"""End-to-end check for the untar wrapper.

Extracts ``kraken2.tar.gz`` and asserts each emitted file's MD5 matches
the expected values.

``test_untar`` is a single ``@env.task``: it fetches its fixture, calls
``untar``, and asserts the extracted files' MD5s. ``main`` runs it once
and waits.

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_untar.py            # local
    uv run --project plugins/bio python plugins/bio/tests/test_untar.py remote     # devbox / prod
"""

from __future__ import annotations

import asyncio

import flyte
from _utils import FileT, assert_md5_files, cli_mode, fixture_file, init_for_mode

from flyteplugins.bio.untar import env as untar_env
from flyteplugins.bio.untar import untar

env = flyte.TaskEnvironment(name="untar_tests", depends_on=[untar_env])


@env.task
async def test_untar() -> None:
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


async def main() -> None:
    mode = cli_mode()
    await init_for_mode(mode)
    runner = flyte.with_runcontext(mode=mode)
    run = await runner.run.aio(test_untar)
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
