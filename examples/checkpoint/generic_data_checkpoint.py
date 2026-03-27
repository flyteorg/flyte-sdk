# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
# ]
# ///

"""
Generic JSON checkpoint
=======================

Uses :attr:`flyte.ctx().checkpoint` with a single JSON file under the local
checkpoint workspace. After :meth:`~flyte.AsyncCheckpoint.load` or
``await checkpoint.load.aio()``, files may
sit under a subdirectory (depending on the remote prefix); ``rglob`` finds
``state.json`` reliably.

**Note:** Use an editable install of this SDK (or a release that includes
``TaskContext.checkpoint``) when running examples; a bare ``uv run`` may resolve
an older ``flyte`` from PyPI.
"""

from __future__ import annotations

import json
import pathlib

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_generic_json",
    image=flyte.Image.from_debian_base(),
)

STATE_FILE = "state.json"
RETRIES = 3


def resolve_state_file(root: pathlib.Path) -> pathlib.Path:
    direct = root / STATE_FILE
    if direct.exists():
        return direct
    matches = list(root.rglob(STATE_FILE))
    if matches:
        return matches[0]
    return direct


@env.task(retries=RETRIES)
async def durable_counter(steps: int = 10) -> int:
    tctx = flyte.ctx()
    assert tctx is not None
    ck = tctx.checkpoint
    assert ck is not None

    await ck.load.aio()
    path = resolve_state_file(ck.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    print("PATH", path)
    print("REMOTE PATH", ck._checkpoint_dest)
    if path.exists():
        start = int(json.loads(path.read_text(encoding="utf-8"))["index"])

    print("START", start)

    for index in range(start, steps):
        if index == (steps // RETRIES):
            raise RuntimeError("Simulated failure")

        path.write_text(json.dumps({"index": index}), encoding="utf-8")
        await ck.save.aio(local_path=path)
    return index


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote").run(durable_counter, steps=3)
    print(run.url)
