"""
Upload files with a content hash computed while streaming, from both sync and async tasks.

Covers the `File.from_local_sync` / `File.from_local` upload paths that route through the
obstore-aware storage layer (auto-sized multipart parts), with and without a `HashMethod`.

Run with:

```
flyte run examples/basics/file_hash_upload.py main
```
"""

import hashlib
import os
import tempfile

import flyte
from flyte.io import File
from flyte.io._hashing_io import HashlibAccumulator, PrecomputedValue

env = flyte.TaskEnvironment("file_hash_upload")


def _write_local(size_bytes: int) -> tuple[str, str]:
    """Write `size_bytes` of pseudo-random data to a temp file; return (path, sha256)."""
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as tmp:
        remaining = size_bytes
        while remaining > 0:
            chunk = os.urandom(min(1 << 20, remaining))
            tmp.write(chunk)
            h.update(chunk)
            remaining -= len(chunk)
        return tmp.name, h.hexdigest()


@env.task
def upload_sync_no_hash(size_mib: int) -> File:
    """Sync upload, no hash: from_local_sync -> syncify(storage.put)."""
    path, _ = _write_local(size_mib << 20)
    try:
        f = File.from_local_sync(path)
        print(f"[sync/no-hash] {path} -> {f.path} (hash={f.hash})")
        assert f.hash is None
        return f
    finally:
        os.unlink(path)


@env.task
def upload_sync_with_hash(size_mib: int) -> tuple[File, str]:
    """Sync upload with a streaming sha256: from_local_sync -> syncify(_upload_hashed) -> put_stream(size_hint)."""
    path, expected = _write_local(size_mib << 20)
    try:
        f = File.from_local_sync(path, hash_method=HashlibAccumulator.from_hash_name("sha256"))
        print(f"[sync/hash] {path} -> {f.path}\n  streamed={f.hash}\n  expected={expected}")
        assert f.hash == expected, "streamed hash must match independently computed sha256"
        return f, expected
    finally:
        os.unlink(path)


@env.task
def upload_sync_precomputed(size_mib: int) -> File:
    """Sync upload with a precomputed hash: skips hashing, still goes through storage.put."""
    path, expected = _write_local(size_mib << 20)
    try:
        f = File.from_local_sync(path, hash_method=PrecomputedValue(expected))
        print(f"[sync/precomputed] {path} -> {f.path} (hash={f.hash})")
        assert f.hash == expected
        return f
    finally:
        os.unlink(path)


@env.task
async def upload_async_with_hash(size_mib: int) -> tuple[File, str]:
    """Async upload with a streaming sha256: from_local -> _upload_hashed -> put_stream(size_hint)."""
    path, expected = _write_local(size_mib << 20)
    try:
        f = await File.from_local(path, hash_method=HashlibAccumulator.from_hash_name("sha256"))
        print(f"[async/hash] {path} -> {f.path}\n  streamed={f.hash}\n  expected={expected}")
        assert f.hash == expected, "streamed hash must match independently computed sha256"
        return f, expected
    finally:
        os.unlink(path)


@env.task
def verify_remote(f: File, expected_sha256: str, expected_size: int) -> bool:
    """Download the uploaded object (download_sync) and confirm size + sha256 survived the round trip."""
    local = f.download_sync()
    h = hashlib.sha256()
    size = 0
    with open(local, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    ok = h.hexdigest() == expected_sha256 and size == expected_size
    print(f"[verify] {f.path}: size={size} sha256={h.hexdigest()} ok={ok}")
    assert ok, f"round-trip mismatch for {f.path}"
    return ok


@env.task
async def main(size_mib: int = 64) -> list[bool]:
    """Exercise every File upload path and verify each object end to end."""
    expected_size = size_mib << 20

    # Sync tasks called from an async task must go through `.aio` (see #1472).
    f1 = await upload_sync_no_hash.aio(size_mib)
    f2, h2 = await upload_sync_with_hash.aio(size_mib)
    f3 = await upload_sync_precomputed.aio(size_mib)
    f4, h4 = await upload_async_with_hash(size_mib)

    results = [
        await verify_remote.aio(f2, h2, expected_size),
        await verify_remote.aio(f3, f3.hash or "", expected_size),
        await verify_remote.aio(f4, h4, expected_size),
    ]
    # f1 has no hash; just confirm it exists remotely with the right size.
    local = await f1.download()
    results.append(os.path.getsize(local) == expected_size)
    print(f"all ok: {all(results)}")
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
