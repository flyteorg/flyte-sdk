"""Shared helpers for the bio plugin's end-to-end tests.

Each bio task wraps a tool that already has a corresponding nf-core
module, and we lean on the same test data those modules use:
nf-core/test-datasets (branch ``modules``).
"""

from __future__ import annotations

import gzip
import hashlib
import pathlib
import sys
import urllib.request
from typing import Any, Literal, cast

from _constants import CACHE_DIR, NF_CORE_RAW_BASE
from flyte.io import File

Mode = Literal["local", "remote"]


def cli_mode(argv: list[str] | None = None) -> Mode:
    """Pick ``"local"`` or ``"remote"`` from the test script's argv.

    Default is ``"local"`` (in-process execution). Pass ``remote`` as the
    first script argument to submit each run through a real Flyte
    cluster.

    This is **not** redundant with the config: ``flyte.init_from_config``
    walks a precedence list and will silently pick up any
    ``~/.union/config.yaml`` or ``~/.flyte/config.yaml`` that exists.
    When such a config has an endpoint, ``with_runcontext()`` (no
    explicit mode) auto-flips to remote — which is surprising for a test
    you expect to run locally. Forcing ``mode="local"`` here makes the
    default predictable; pass ``remote`` to opt in.
    """
    argv = argv if argv is not None else sys.argv
    if len(argv) > 1 and argv[1] == "remote":
        return "remote"
    return "local"


# Convenience alias — File is generic; tests rarely care about the
# subtype, and writing ``File[Any]`` at each call site is noisy.
FileT = File[Any]


def nf_core_fixture(relative_path: str) -> pathlib.Path:
    """Return a local path to an nf-core/test-datasets fixture, fetching on miss.

    ``relative_path`` is taken relative to the ``data/`` root of the
    ``modules`` branch — e.g. ``"genomics/sarscov2/genome/bed/test.bed"``.
    """
    dest = CACHE_DIR / relative_path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{NF_CORE_RAW_BASE}/{relative_path}", dest)
    return dest


async def nf_core_file(relative_path: str) -> FileT:
    """Fetch an nf-core fixture and wrap it as a :class:`flyte.io.File`."""
    return cast(FileT, await File.from_local(str(nf_core_fixture(relative_path))))


def md5(path: pathlib.Path) -> str:
    """MD5 of a file's content, with gzip auto-decompression.

    nf-test computes snapshot MD5s against the *decompressed* content of
    ``.gz`` / ``.bgz`` outputs, not the raw bytes. We match that so
    snapshot values from nf-core's ``main.nf.test.snap`` can be pasted
    in unchanged — gzip framing (mtime, OS byte, compression level)
    isn't reproducible across containers and would otherwise force a
    re-snapshot every time the upstream image is bumped.
    """
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return hashlib.md5(data).hexdigest()


async def assert_md5(label: str, out: FileT, expected: str) -> None:
    """Download ``out`` and assert its MD5 matches ``expected``.

    Failure message includes the full output body so divergences from
    upstream snapshots are easy to diff by eye.
    """
    local = pathlib.Path(await out.download())
    actual = md5(local)
    assert actual == expected, (
        f"{label}: output diverges from nf-core snapshot\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"  body: {local.read_bytes().decode(errors='replace')!r}"  # noqa: ASYNC240 — small local file, failure path only
    )
    print(f"ok: {label} md5={actual}")


async def assert_md5_files(
    label: str,
    files: list[FileT],
    expected: dict[str, str],
) -> None:
    """Assert each file's MD5 matches ``expected[basename]``.

    For tasks with multi-file outputs (e.g. ``Glob`` declarations) where
    ordering isn't meaningful. Diffs file sets *and* MD5s so a failure
    pinpoints whether the wrapper produced extra/missing files or just
    one with the wrong content.
    """
    actual: dict[str, str] = {}
    for f in files:
        local = pathlib.Path(await f.download())
        actual[local.name] = md5(local)
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    mismatched = {k: (actual[k], expected[k]) for k in set(expected) & set(actual) if actual[k] != expected[k]}
    assert not (missing or extra or mismatched), (
        f"{label}: outputs diverge from nf-core snapshot\n"
        f"  expected files: {sorted(expected)}\n"
        f"  actual files:   {sorted(actual)}\n"
        f"  missing: {sorted(missing)}\n"
        f"  extra:   {sorted(extra)}\n"
        f"  mismatched md5: {mismatched}"
    )
    print(f"ok: {label} ({len(files)} files)")


