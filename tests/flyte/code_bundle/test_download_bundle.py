"""Tests for download_bundle's concurrency-safe download + extract.

Multiple worker processes (e.g. the torchrun-spawned ranks of a clustered task) call
download_bundle concurrently against the same destination; these tests cover the flock + marker
serialization that keeps them from racing on the tgz download and tar extraction.
"""

import asyncio
import concurrent.futures
import multiprocessing
import os
import pathlib
import sys
import tarfile

import pytest

import flyte.storage
from flyte._code_bundle import bundle as bundle_module
from flyte._code_bundle.bundle import download_bundle
from flyte.models import CodeBundle

BUNDLE_NAME = "fastdeadbeef.tar.gz"
BUNDLE_FILES = {"pkg/mod_a.py": b"A = 1\n", "pkg/sub/mod_b.py": b"B = 2\n", "main.py": b"import pkg\n"}

needs_flock = pytest.mark.skipif(sys.platform == "win32", reason="requires fcntl.flock")


@pytest.fixture
def tarball(tmp_path_factory) -> pathlib.Path:
    src = tmp_path_factory.mktemp("bundle_src")
    for rel, content in BUNDLE_FILES.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    archive = tmp_path_factory.mktemp("bundle_tgz") / BUNDLE_NAME
    with tarfile.open(archive, "w:gz") as tar:
        for rel in BUNDLE_FILES:
            tar.add(src / rel, arcname=rel)
    return archive


def make_bundle(dest: pathlib.Path, name: str = BUNDLE_NAME, pkl: bool = False) -> CodeBundle:
    kwargs = {"pkl" if pkl else "tgz": f"s3://bucket/{name}"}
    return CodeBundle(computed_version="v1", destination=str(dest), **kwargs)


class CopyingGet:
    """Stand-in for flyte.storage.get that copies a local fixture and counts invocations."""

    def __init__(self, source: pathlib.Path, delay: float = 0.0):
        self.source = source
        self.delay = delay
        self.calls = 0

    async def __call__(self, remote: str, local: str, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        pathlib.Path(local).write_bytes(self.source.read_bytes())


def assert_extracted(dest: pathlib.Path):
    for rel, content in BUNDLE_FILES.items():
        assert (dest / rel).read_bytes() == content


@pytest.mark.asyncio
async def test_happy_path_tgz(tmp_path, tarball, monkeypatch):
    fake = CopyingGet(tarball)
    monkeypatch.setattr(flyte.storage, "get", fake)

    result = await download_bundle(make_bundle(tmp_path))

    assert result == (tmp_path / BUNDLE_NAME).absolute()
    assert fake.calls == 1
    assert_extracted(tmp_path)
    assert (tmp_path / f".{BUNDLE_NAME}.extracted").exists()
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_marker_fast_path_skips_download(tmp_path, tarball, monkeypatch):
    fake = CopyingGet(tarball)
    monkeypatch.setattr(flyte.storage, "get", fake)
    (tmp_path / BUNDLE_NAME).touch()
    (tmp_path / f".{BUNDLE_NAME}.extracted").touch()

    result = await download_bundle(make_bundle(tmp_path))

    assert result == (tmp_path / BUNDLE_NAME).absolute()
    assert fake.calls == 0


@needs_flock
@pytest.mark.asyncio
async def test_waiter_sees_marker_written_by_lock_holder(tmp_path, tarball, monkeypatch):
    import fcntl

    fake = CopyingGet(tarball)
    monkeypatch.setattr(flyte.storage, "get", fake)

    fd = os.open(tmp_path / f".{BUNDLE_NAME}.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        task = asyncio.create_task(download_bundle(make_bundle(tmp_path)))
        await asyncio.sleep(0.3)
        assert not task.done()
        (tmp_path / BUNDLE_NAME).touch()
        (tmp_path / f".{BUNDLE_NAME}.extracted").touch()
    finally:
        os.close(fd)

    result = await asyncio.wait_for(task, timeout=5)
    assert result == (tmp_path / BUNDLE_NAME).absolute()
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_stale_tgz_without_marker_is_redownloaded(tmp_path, tarball, monkeypatch):
    fake = CopyingGet(tarball)
    monkeypatch.setattr(flyte.storage, "get", fake)
    (tmp_path / BUNDLE_NAME).write_bytes(tarball.read_bytes()[: tarball.stat().st_size // 2])

    await download_bundle(make_bundle(tmp_path))

    assert fake.calls == 1
    assert_extracted(tmp_path)
    assert (tmp_path / f".{BUNDLE_NAME}.extracted").exists()


@pytest.mark.asyncio
async def test_corrupt_bundle_leaves_no_marker_and_retry_succeeds(tmp_path, tarball, monkeypatch):
    class GarbageGet(CopyingGet):
        async def __call__(self, remote: str, local: str, **kwargs):
            self.calls += 1
            pathlib.Path(local).write_bytes(b"not a tarball")

    monkeypatch.setattr(flyte.storage, "get", GarbageGet(tarball))
    with pytest.raises(RuntimeError):
        await download_bundle(make_bundle(tmp_path))
    assert not (tmp_path / f".{BUNDLE_NAME}.extracted").exists()

    monkeypatch.setattr(flyte.storage, "get", CopyingGet(tarball))
    await download_bundle(make_bundle(tmp_path))
    assert_extracted(tmp_path)


@needs_flock
@pytest.mark.asyncio
async def test_same_process_concurrency_downloads_once(tmp_path, tarball, monkeypatch):
    fake = CopyingGet(tarball, delay=0.2)
    monkeypatch.setattr(flyte.storage, "get", fake)

    results = await asyncio.gather(*[download_bundle(make_bundle(tmp_path)) for _ in range(8)])

    assert all(r == (tmp_path / BUNDLE_NAME).absolute() for r in results)
    assert fake.calls == 1
    assert_extracted(tmp_path)


def _worker(dest: str, archive: str, sentinel_dir: str, idx: int) -> str:
    """Child-process entrypoint: patch storage.get with a slow copy that drops a sentinel."""
    import flyte.storage as storage
    from flyte._code_bundle.bundle import download_bundle as dl

    async def slow_get(remote: str, local: str, **kwargs):
        pathlib.Path(sentinel_dir, f"downloaded-by-{idx}").touch()
        await asyncio.sleep(0.3)
        pathlib.Path(local).write_bytes(pathlib.Path(archive).read_bytes())

    storage.get = slow_get
    result = asyncio.run(dl(CodeBundle(computed_version="v1", destination=dest, tgz=f"s3://bucket/{BUNDLE_NAME}")))
    return str(result)


@needs_flock
def test_multi_process_concurrency(tmp_path, tarball):
    """The production repro: N processes download+extract the same bundle into one directory."""
    sentinels = tmp_path / "sentinels"
    sentinels.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = [pool.submit(_worker, str(dest), str(tarball), str(sentinels), i) for i in range(4)]
        results = [f.result(timeout=120) for f in futures]

    assert all(r == str((dest / BUNDLE_NAME).absolute()) for r in results)
    assert len(list(sentinels.iterdir())) == 1
    assert_extracted(dest)
    assert (dest / f".{BUNDLE_NAME}.extracted").exists()


@pytest.mark.asyncio
async def test_degraded_mode_without_fcntl(tmp_path, tarball, monkeypatch):
    monkeypatch.setattr(bundle_module, "fcntl", None)
    fake = CopyingGet(tarball)
    monkeypatch.setattr(flyte.storage, "get", fake)

    # Legacy semantics: a pre-existing tgz short-circuits, marker not required.
    (tmp_path / BUNDLE_NAME).touch()
    result = await download_bundle(make_bundle(tmp_path))
    assert result == (tmp_path / BUNDLE_NAME).absolute()
    assert fake.calls == 0

    # And a fresh dest still downloads + extracts.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    await download_bundle(make_bundle(fresh))
    assert fake.calls == 1
    assert_extracted(fresh)


@pytest.mark.asyncio
async def test_marker_write_failure_is_non_fatal(tmp_path, tarball, monkeypatch):
    monkeypatch.setattr(flyte.storage, "get", CopyingGet(tarball))

    def failing_touch(self, *args, **kwargs):
        if self.name.endswith(".extracted"):
            raise OSError("disk full")
        return original_touch(self, *args, **kwargs)

    original_touch = pathlib.Path.touch
    monkeypatch.setattr(pathlib.Path, "touch", failing_touch)

    result = await download_bundle(make_bundle(tmp_path))
    assert result == (tmp_path / BUNDLE_NAME).absolute()
    assert_extracted(tmp_path)


@pytest.mark.asyncio
async def test_pkl_branch(tmp_path, tarball, monkeypatch):
    name = "fastcafe.pkl.gz"
    fake = CopyingGet(tarball, delay=0.1)
    monkeypatch.setattr(flyte.storage, "get", fake)

    results = await asyncio.gather(*[download_bundle(make_bundle(tmp_path, name=name, pkl=True)) for _ in range(4)])

    assert all(r == (tmp_path / name).absolute() for r in results)
    assert fake.calls == 1
    assert (tmp_path / f".{name}.done").exists()
    assert not list(tmp_path.glob("*.part"))

    # Fast path on a subsequent call.
    await download_bundle(make_bundle(tmp_path, name=name, pkl=True))
    assert fake.calls == 1


@needs_flock
@pytest.mark.asyncio
async def test_different_bundles_do_not_contend(tmp_path, tarball, monkeypatch):
    fake = CopyingGet(tarball, delay=0.2)
    monkeypatch.setattr(flyte.storage, "get", fake)
    other = "fastfeedface.tar.gz"

    results = await asyncio.wait_for(
        asyncio.gather(download_bundle(make_bundle(tmp_path)), download_bundle(make_bundle(tmp_path, name=other))),
        timeout=10,
    )

    assert fake.calls == 2
    assert {r.name for r in results} == {BUNDLE_NAME, other}
