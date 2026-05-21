"""Local (tar-archive) Volume backend.

A ``Volume`` with ``volume_backend="local"`` is materialized as a single
``tar.gz`` archive in object storage. ``mount()`` extracts the archive
into ``mount_path``; ``commit()`` tars ``mount_path`` and uploads the new
archive; ``fork()`` is a server-side copy of the existing archive via
:meth:`flyte.io.File.copy_to`.

Unlike the JuiceFS backend, this one requires no FUSE, no daemon, no
privileged container, and no special pod template — anything that can
read/write a directory and shell out to ``tar`` works. The trade-off is
that every commit re-tars the entire tree (no chunk dedup, no incremental
upload), so it's only practical for ≤ a few GB.

The same code path works in ``flyte run --local`` because the underlying
``flyte.storage`` / ``fsspec`` layer transparently maps remote URIs onto
local paths when the raw-data root is local.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from flyte._context import internal_ctx
from flyte._logging import logger as _logger
from flyte.io._file import File
from flyte.io._hashing_io import HashlibAccumulator

if TYPE_CHECKING:
    from flyte.extras._volume import Volume

_ARCHIVE_NAME = "volume.tar.gz"


async def local_mount(
    volume: "Volume",
    *,
    mount_path: str,
) -> None:
    """Materialize ``volume`` at ``mount_path`` for the local backend.

    For a fresh Volume (``index is None``), creates an empty directory.
    For a committed Volume, downloads the archive and extracts it.
    Refuses to extract over a non-empty ``mount_path`` to match the
    JuiceFS "format over non-empty bucket" behavior.
    """
    mp = Path(mount_path)
    mp.mkdir(parents=True, exist_ok=True)

    if volume.index is None:
        _logger.info("[Volume.mount(local)] fresh mount at %s", mount_path)
        return

    if any(mp.iterdir()):
        raise RuntimeError(
            f"Cannot mount local Volume at {mount_path!r}: directory is not empty. "
            "Local-backend mounts require an empty mount point so the archive "
            "can be extracted without colliding with pre-existing files."
        )

    archive_path = await volume.index.download()
    try:
        _logger.info("[Volume.mount(local)] extracting %s -> %s", volume.index.path, mount_path)
        await asyncio.to_thread(
            subprocess.run,
            ["tar", "-xzf", archive_path, "-C", mount_path],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.unlink(archive_path)
        except OSError:
            pass


async def local_commit(
    volume: "Volume",
    *,
    mount_path: str,
) -> "Volume":
    """Tar ``mount_path`` and upload as the new ``index``."""
    return await _tar_and_upload(volume, mount_path=mount_path, parent_index=volume.index)


async def local_commit_inplace(
    volume: "Volume",
    *,
    mount_path: str,
) -> "Volume":
    """Snapshot ``mount_path`` to a fresh archive without unmounting.

    Identical to :func:`local_commit` for the local backend since there's
    no live daemon to keep running — ``mount_path`` is just a directory.
    """
    return await _tar_and_upload(volume, mount_path=mount_path, parent_index=volume.index)


async def local_fork(
    volume: "Volume",
    name: str,
) -> "Volume":
    """Server-side copy of the current ``index`` to a new archive URI."""
    if volume.index is None:
        raise RuntimeError("Cannot fork: Volume has no committed index. Commit the parent first, then fork.")
    ctx = internal_ctx()
    dst_path = ctx.raw_data.get_random_remote_path(file_name=_ARCHIVE_NAME)
    _logger.info("[Volume.fork(local)] copying %s -> %s", volume.index.path, dst_path)
    new_index = await volume.index.copy_to(dst_path)
    return volume.__class__(
        name=name,
        bucket=volume.bucket,
        storage=volume.storage,
        index=new_index,
        parent=volume.index,
        metadata_engine=volume.metadata_engine,
        volume_backend="local",
        used_bytes=volume.used_bytes,
        inode_count=volume.inode_count,
        index_bytes=volume.index_bytes,
    )


async def _tar_and_upload(
    volume: "Volume",
    *,
    mount_path: str,
    parent_index: Optional[File],
) -> "Volume":
    """Tar ``mount_path`` to a temp file, upload, and return a new Volume."""
    with tempfile.NamedTemporaryFile(prefix="vol-local-", suffix=".tar.gz", delete=False) as f:
        archive = f.name

    try:
        await asyncio.to_thread(os.sync)
        _logger.info("[Volume.commit(local)] taring %s -> %s", mount_path, archive)
        await asyncio.to_thread(
            subprocess.run,
            ["tar", "-czf", archive, "-C", mount_path, "."],
            check=True,
            capture_output=True,
            text=True,
        )
        index_bytes = os.path.getsize(archive)
        used_bytes, inode_count = await asyncio.to_thread(_walk_stats, mount_path)
        ctx = internal_ctx()
        new_index: File = await File.from_local(
            archive,
            remote_destination=ctx.raw_data.get_random_remote_path(file_name=_ARCHIVE_NAME),
            hash_method=HashlibAccumulator.from_hash_name("md5"),
        )
        return volume.__class__(
            name=volume.name,
            bucket=volume.bucket,
            storage=volume.storage,
            index=new_index,
            parent=parent_index,
            metadata_engine=volume.metadata_engine,
            volume_backend="local",
            used_bytes=used_bytes,
            inode_count=inode_count,
            index_bytes=index_bytes,
        )
    finally:
        try:
            os.unlink(archive)
        except OSError:
            pass


def _walk_stats(root: str) -> Tuple[int, int]:
    """Walk ``root`` and return ``(total_bytes, inode_count)``.

    Counts files, dirs, and symlinks. Skips files we can't stat (e.g.
    permission errors), since stats are best-effort.
    """
    total_bytes = 0
    inode_count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        inode_count += len(dirnames) + len(filenames)
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            total_bytes += st.st_size
    return total_bytes, inode_count


def _rm_mount(mount_path: str) -> None:
    """Best-effort cleanup of a local-backend mount directory."""
    try:
        shutil.rmtree(mount_path)
    except OSError as e:
        _logger.warning("[Volume.local] could not remove %s: %s", mount_path, e)
