"""Shared helpers for the bio plugin's end-to-end tests.

The tests use a shared public fixture set and compare outputs against
checked-in expected MD5 values.
"""

from __future__ import annotations

import gzip
import hashlib
import pathlib
import urllib.request
from typing import Any, cast

import flyte
from _constants import CACHE_DIR, FIXTURE_BASE_URL
from flyte.io import File

# Convenience alias — File is generic; tests rarely care about the
# subtype, and writing ``File[Any]`` at each call site is noisy.
FileT = File[Any]


def init_local_flyte() -> None:
    """Initialize Flyte for local pytest runs."""
    flyte.init()


async def run_local_task(task: Any, **kwargs: Any) -> None:
    """Run a Flyte task locally and wait for completion."""
    run = await flyte.with_runcontext(mode="local").run.aio(task, **kwargs)
    await run.wait.aio()


def fixture_path(relative_path: str) -> pathlib.Path:
    """Return a local path to a test fixture, fetching on miss.

    ``relative_path`` is taken relative to the shared fixture root, for
    example ``"genomics/sarscov2/genome/bed/test.bed"``.
    """
    dest = CACHE_DIR / relative_path
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{FIXTURE_BASE_URL}/{relative_path}", dest)
    return dest


async def fixture_file(relative_path: str) -> FileT:
    """Fetch a fixture and wrap it as a :class:`flyte.io.File`."""
    return cast(FileT, await File.from_local(str(fixture_path(relative_path))))


def md5(path: pathlib.Path) -> str:
    """MD5 of a file's content, with gzip auto-decompression.

    The expected checksums are computed against the *decompressed*
    content of ``.gz`` / ``.bgz`` outputs, not the raw bytes. That keeps
    the assertions stable even when gzip framing differs across
    environments.
    """
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return hashlib.md5(data).hexdigest()


async def assert_md5(label: str, out: FileT, expected: str) -> None:
    """Download ``out`` and assert its MD5 matches ``expected``.

    Failure message includes the full output body so divergences are easy
    to diff by eye.
    """
    local = pathlib.Path(await out.download())
    actual = md5(local)
    assert actual == expected, (
        f"{label}: output did not match expected checksum\n"
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
        f"{label}: outputs did not match expected checksums\n"
        f"  expected files: {sorted(expected)}\n"
        f"  actual files:   {sorted(actual)}\n"
        f"  missing: {sorted(missing)}\n"
        f"  extra:   {sorted(extra)}\n"
        f"  mismatched md5: {mismatched}"
    )
    print(f"ok: {label} ({len(files)} files)")
