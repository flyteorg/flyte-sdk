"""Run every bio plugin end-to-end test as children of a SINGLE run.

Each module exposes one top-level ``test_<tool>`` task; this harness
imports them and fans them out under one parent task, so the whole suite
is one command and one run tree (parent → per-tool → per-scenario →
shell task). ``test_<tool>`` asserts inline, so a failure surfaces when
the run reaches a terminal state.

Run it::

    uv run --project plugins/bio python plugins/bio/tests/test_all.py            # local
    uv run --project plugins/bio python plugins/bio/tests/test_all.py remote     # devbox / prod
"""

from __future__ import annotations

import asyncio

import flyte
from _utils import cli_mode, init_for_mode
from test_bedtools import env as bedtools_tests_env
from test_bedtools import test_bedtools
from test_cat import env as cat_tests_env
from test_cat import test_cat
from test_gunzip import env as gunzip_tests_env
from test_gunzip import test_gunzip
from test_untar import env as untar_tests_env
from test_untar import test_untar

env = flyte.TaskEnvironment(
    name="all_tests",
    depends_on=[bedtools_tests_env, cat_tests_env, gunzip_tests_env, untar_tests_env],
)


@env.task
async def test_all() -> None:
    """Fan out every tool's test suite as children of one run."""
    await asyncio.gather(
        test_bedtools(),
        test_cat(),
        test_gunzip(),
        test_untar(),
    )


async def main() -> None:
    mode = cli_mode()
    await init_for_mode(mode)
    runner = flyte.with_runcontext(mode=mode)
    run = await runner.run.aio(test_all)
    print(f"Run URL: {run.url}")
    await run.wait.aio()


if __name__ == "__main__":
    asyncio.run(main())
