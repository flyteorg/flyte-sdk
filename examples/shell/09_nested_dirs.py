"""Nested Dir output from a ContainerTask.

The container writes a directory tree into its ``Dir`` output: a root file
plus two subdirectories, where a filename (``dup.txt``) collides across the two
subdirs.
"""

import os
import sys
import tempfile

import aiofiles

import flyte
from flyte.extras import shell
from flyte.io import Dir

EXPECTED = {
    "root.txt": "root",
    "nested/deep.txt": "deep",
    "nested/dup.txt": "dup-in-nested",
    "other/dup.txt": "dup-in-other",
}

make_nested = shell.create(
    name="make_nested_dir",
    image="debian:12-slim",
    outputs={"out": Dir},
    script=r"""
        mkdir -p {outputs.out}/nested {outputs.out}/other
        printf 'root'          > {outputs.out}/root.txt
        printf 'deep'          > {outputs.out}/nested/deep.txt
        printf 'dup-in-nested' > {outputs.out}/nested/dup.txt
        printf 'dup-in-other'  > {outputs.out}/other/dup.txt
    """,
    cache="disable",
)

env = flyte.TaskEnvironment(name="nested_dirs", depends_on=[make_nested.env])


@env.task
async def verify_nested() -> str:
    """Produce a nested Dir via the container task, download it, assert structure."""
    d: Dir = await make_nested()

    local = await d.download(tempfile.mkdtemp())

    found: dict[str, str] = {}
    for parent, _, files in os.walk(local):
        for f in files:
            full = os.path.join(parent, f)
            rel = os.path.relpath(full, local).replace(os.sep, "/")
            async with aiofiles.open(full) as fh:
                found[rel] = await fh.read()

    problems = []
    for rel, body in EXPECTED.items():
        if rel not in found:
            problems.append(f"MISSING {rel}")
        elif found[rel] != body:
            problems.append(f"WRONG   {rel}: expected {body!r}, got {found[rel]!r}")
    extra = sorted(set(found) - set(EXPECTED))
    if extra:
        problems.append(f"EXTRA   {extra}")

    if problems:
        raise AssertionError(
            "Nested Dir round-trip FAILED (the fix is not active):\n  " + "\n  ".join(problems) + f"\n  found = {found}"
        )

    return f"OK — {len(found)} files, nesting preserved: {sorted(found)}"


if __name__ == "__main__":
    flyte.init_from_config()
    mode = "remote" if (len(sys.argv) > 1 and sys.argv[1] == "remote") else "local"
    run = flyte.with_runcontext(mode=mode).run(verify_nested)
    print(run.url if mode == "remote" else run)
    print(run.outputs())
