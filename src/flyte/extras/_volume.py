"""
Persistent ``Volume`` type and pod-template helper.

A ``Volume`` is a Flyte SDK type that materializes as a mountable filesystem
inside a task pod. Its identity is carried by a single SQLite "index" file
plus a pointer to the object-store bucket holding the immutable data chunks.
Because the index is just a file, ``Volume`` rides through the normal Flyte
literal system as a Pydantic model containing a ``flyte.io.File``, which
means lineage, caching, fork, and clone are first-class.

Typical usage::

    from flyte.extras import Volume, volume_pod_template, volume_image

    env = flyte.TaskEnvironment(
        name="vol-demo",
        pod_template=volume_pod_template(),
        image=volume_image(my_base_image),
    )

    @env.task
    async def write(vol: Volume) -> Volume:
        await vol.mount()
        Path("/workspace/hello.txt").write_text("hi")
        return await vol.commit()

The current implementation runs the filesystem client in-process inside the
primary container (the runtime is layered onto the user image by
:func:`volume_image`), so no sidecar or mount-propagation configuration is
needed. ``mount()`` spawns the client as a subprocess and tracks its PID;
``commit()`` requests a clean unmount and waits for metadata to be flushed
to the index file before uploading.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import ClassVar, Dict, Optional, Set

from pydantic import BaseModel, ConfigDict

from flyte._context import internal_ctx
from flyte._image import Image
from flyte._logging import logger as _logger
from flyte._pod import PodTemplate
from flyte.io._file import File

_DEFAULT_META_DIR = "/var/lib/flyte-volume"
_DEFAULT_CACHE_DIR = "/var/cache/flyte-volume"
_DEFAULT_MOUNT_PATH = "/workspace"
_SQLITE_INDEX_FILENAME = "index.db"
_REDIS_INDEX_FILENAME = "dump.rdb"
_CLIENT_BINARY = "juicefs"  # implementation detail
# Pinned to a release whose SQLite ``jfs_counter`` and Redis ``next{chunk,inode,session}``
# keys are known to match what :func:`_disjoint_fork_counters` expects.
# Verified against 1.1.2 and 1.3.1 — bumping further requires re-validating
# the counter schema (see tests/flyte/extras/test_volume.py::TestForkIsolationRegression).
_CLIENT_VERSION = "1.3.1"
_REDIS_PORT = 6379


def _index_filename(engine: str) -> str:
    return _REDIS_INDEX_FILENAME if engine == "redis" else _SQLITE_INDEX_FILENAME


def _meta_url(meta_dir: str, engine: str) -> str:
    if engine == "redis":
        return f"redis://127.0.0.1:{_REDIS_PORT}/0"
    return f"sqlite3://{Path(meta_dir) / _SQLITE_INDEX_FILENAME}"


class Volume(BaseModel):
    """A persistent volume identified by its metadata index.

    A ``Volume`` is content-addressable lineage on top of an object-store
    bucket. The bucket holds the data chunks; the index (a SQLite ``.db``
    or a Redis ``dump.rdb`` snapshot, depending on ``metadata_engine``)
    holds the entire namespace. Cloning the index produces an independent
    fork that initially sees the same file tree and shares chunk objects
    but diverges as either side writes — see :meth:`fork` for the chunk-key
    disjointness guarantees that make concurrent writes safe.
    """

    name: str
    bucket: str
    storage: str = "s3"
    index: Optional[File] = None
    parent: Optional[File] = None
    # Metadata engine used at runtime. ``None`` resolves to ``"sqlite"`` for
    # backward compatibility with Volumes serialized before this field existed.
    # ``Volume.empty()`` sets this to ``"redis"`` on new volumes.
    metadata_engine: Optional[str] = None
    # Snapshot of metadata stats captured at commit() / fork() time. Best-effort
    # — left as None if the underlying status query failed. Sourced from
    # ``juicefs status`` (``Statistic.UsedSpace`` / ``Statistic.UsedInodes``);
    # inode_count counts files + dirs + symlinks combined.
    used_bytes: Optional[int] = None
    inode_count: Optional[int] = None
    # Size in bytes of the published metadata index (SQLite ``.db`` or
    # Redis ``dump.rdb``). Populated at commit() / fork() time from the
    # on-disk snapshot just before upload; ``None`` for un-committed Volumes.
    index_bytes: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Track active mounts so commit() can wait on the daemon.
    _live_procs: ClassVar[Dict[str, subprocess.Popen]] = {}
    # Track in-process redis-server daemons, keyed by meta_dir.
    _live_redis: ClassVar[Dict[str, subprocess.Popen]] = {}
    # meta_dirs that currently have a live engine + daemon. Used by fork()
    # to decide whether to use the live flush-and-upload path or the cold
    # File.copy_to path. Populated by mount(), cleared by commit().
    _live_meta: ClassVar[Set[str]] = set()

    def _engine(self) -> str:
        return self.metadata_engine or "sqlite"

    @classmethod
    def empty(
        cls,
        name: str,
        bucket: Optional[str] = None,
        *,
        storage: str = "s3",
        metadata_engine: str = "redis",
    ) -> "Volume":
        """Declare a brand-new volume. The first ``mount()`` call will
        bootstrap the namespace (the underlying client refuses to format
        over a non-empty bucket prefix).

        If ``bucket`` is omitted, it is derived from the currently active
        task context as ``{raw_data_root}/{project}/{domain}/volumes`` —
        following Flyte's own layout for offloaded data. Must be called
        from inside a task in that case.

        ``metadata_engine`` controls the in-pod metadata backend. ``"redis"``
        (default) runs an in-process ``redis-server`` and persists the
        namespace as an RDB snapshot — faster than SQLite for metadata-heavy
        workloads and more compact on disk. ``"sqlite"`` keeps everything
        on local disk with no extra daemon. The choice is baked into the
        Volume and travels with it through lineage; subsequent mounts of
        the same Volume must use the same engine.
        """
        if bucket is None:
            bucket = _default_bucket()
        return cls(
            name=name,
            bucket=bucket,
            storage=storage,
            index=None,
            parent=None,
            metadata_engine=metadata_engine,
        )

    async def mount(
        self,
        *,
        mount_path: str = _DEFAULT_MOUNT_PATH,
        meta_dir: str = _DEFAULT_META_DIR,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        timeout: float = 120.0,
        writeback: bool = True,
        upload_delay: Optional[str] = None,
        max_uploads: int = 50,
        attr_cache: float = 60.0,
        entry_cache: float = 60.0,
        dir_entry_cache: float = 60.0,
    ) -> None:
        """Format (if fresh) and mount the volume at ``mount_path`` in this
        process.

        Call once near the top of a task body before reading or writing under
        ``mount_path``.

        When ``writeback=True`` (default), writes land in the local cache
        directory first and are uploaded asynchronously in the background.
        This decouples write latency from object-store round-trips. The
        pending upload queue is drained on ``commit()``; if the pod dies
        before commit, in-flight chunks are lost — but that's fine because
        the Volume itself is never published in that case.

        ``upload_delay`` (e.g. ``"1h"``, ``"30m"``, ``"5s"``) defers uploads
        by the given duration. Useful for write-scratchy workloads — files
        that are written and then overwritten / deleted within the delay
        window are never uploaded at all. Default (``None``) is no extra
        delay; the background uploader starts as soon as a chunk is written.
        Has no effect without ``writeback=True``.

        ``max_uploads`` caps concurrent S3 PUTs (default 50; underlying
        client default is 20). Bumping helps write-burst phases when the
        chunks are small enough that 20 streams can't saturate the link.

        ``attr_cache`` / ``entry_cache`` / ``dir_entry_cache`` are kernel-side
        TTLs in seconds for file attributes, name-to-inode lookups, and
        directory listings respectively. Defaults are ``60.0`` for all three,
        which collapses stat / getattr / lookup storms by an order of
        magnitude on directory-heavy workloads (Go toolchain, package
        managers, codegen). This is safe because a Volume is single-writer
        for the duration of its mount — no external mutator is supported by
        this mechanism. Concurrent-writer scenarios will be opt-in via a
        separate API when added.
        """
        engine = self._engine()
        meta = Path(meta_dir)
        meta.mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(mount_path).mkdir(parents=True, exist_ok=True)

        index_path = meta / _index_filename(engine)
        if self.index is not None:
            _logger.info("[Volume.mount] downloading index %s", self.index.path)
            await self.index.download(str(index_path))
        elif index_path.exists():
            # Stale index from an earlier mount in the same pod (shouldn't
            # happen in practice; tasks get fresh pods).
            index_path.unlink()

        if engine == "redis":
            # redis-server picks up dump.rdb from the working dir at startup.
            redis_proc = await _start_redis(meta_dir)
            type(self)._live_redis[meta_dir] = redis_proc

        meta_url = _meta_url(meta_dir, engine)
        client_bucket = _client_bucket_uri(self.bucket, self.storage)
        if self.index is None:
            _logger.info(
                "[Volume.mount] formatting fresh volume name=%s bucket=%s storage=%s",
                self.name,
                client_bucket,
                self.storage,
            )
            await asyncio.to_thread(
                _run_check,
                [_CLIENT_BINARY, "format", "--storage", self.storage, "--bucket", client_bucket, meta_url, self.name],
            )

        mount_cmd = [
            _CLIENT_BINARY,
            "mount",
            "--cache-dir",
            cache_dir,
            "--max-uploads",
            str(max_uploads),
            "--attr-cache",
            str(attr_cache),
            "--entry-cache",
            str(entry_cache),
            "--dir-entry-cache",
            str(dir_entry_cache),
        ]
        if writeback:
            mount_cmd.append("--writeback")
            if upload_delay:
                mount_cmd += ["--upload-delay", upload_delay]
        mount_cmd += [meta_url, mount_path]

        _logger.info(
            "[Volume.mount] mounting at %s (writeback=%s upload_delay=%s "
            "max_uploads=%d attr_cache=%s entry_cache=%s dir_entry_cache=%s)",
            mount_path,
            writeback,
            upload_delay,
            max_uploads,
            attr_cache,
            entry_cache,
            dir_entry_cache,
        )
        proc = subprocess.Popen(  # noqa: ASYNC220 - long-lived daemon held in a class-level dict
            mount_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        type(self)._live_procs[mount_path] = proc

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                raise RuntimeError(f"volume client exited prematurely (rc={proc.returncode}): {out}")
            if _is_fuse_mount(mount_path):
                break
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"mount at {mount_path} did not become a FUSE mountpoint within {timeout}s")
            await asyncio.sleep(0.5)

        type(self)._live_meta.add(meta_dir)
        _logger.info("[Volume.mount] %s is now a FUSE mount", mount_path)

    async def commit(
        self,
        *,
        mount_path: str = _DEFAULT_MOUNT_PATH,
        meta_dir: str = _DEFAULT_META_DIR,
        timeout: float = 60.0,
    ) -> "Volume":
        """Flush, unmount, then upload the updated index as a Flyte ``File``
        and return a new ``Volume`` whose ``parent`` points at the previous
        index.
        """
        engine = self._engine()
        # Flush page cache so any unfsync'd writes hit the filesystem first.
        await asyncio.to_thread(os.sync)

        meta = Path(meta_dir)
        index_path = meta / _index_filename(engine)

        # Tell the client to flush and exit cleanly.
        _logger.info("[Volume.commit] unmounting %s", mount_path)
        await asyncio.to_thread(_run_check, [_CLIENT_BINARY, "umount", mount_path])

        proc = type(self)._live_procs.pop(mount_path, None)
        if proc is not None:
            try:
                await asyncio.to_thread(proc.wait, timeout)
            except subprocess.TimeoutExpired:
                _logger.warning("[Volume.commit] client didn't exit in %ss; killing", timeout)
                proc.kill()
                await asyncio.to_thread(proc.wait, 5)

        if engine == "redis":
            # Flush Redis state to dump.rdb, then query stats while the daemon
            # is still alive, then shut it down. SHUTDOWN NOSAVE prevents a
            # second redundant save on exit.
            await asyncio.to_thread(_redis_save)
            used_bytes, inode_count = await _query_volume_stats(meta_dir, engine)
            redis_proc = type(self)._live_redis.pop(meta_dir, None)
            await asyncio.to_thread(_stop_redis, redis_proc, timeout)
        else:
            # Force WAL pages back into the main .db before uploading.
            await asyncio.to_thread(_wal_checkpoint, str(index_path))
            used_bytes, inode_count = await _query_volume_stats(meta_dir, engine)

        type(self)._live_meta.discard(meta_dir)
        ctx = internal_ctx()
        index_bytes = os.path.getsize(str(index_path))
        # No hash_method — see fork()'s upload site for why (streaming-hash
        # path produces thousands of tiny S3 writes on binary indexes).
        new_index: File = await File.from_local(
            str(index_path),
            remote_destination=ctx.raw_data.get_random_remote_path(file_name=_index_filename(engine)),
        )
        return Volume(
            name=self.name,
            bucket=self.bucket,
            storage=self.storage,
            index=new_index,
            parent=self.index,
            metadata_engine=self.metadata_engine,
            used_bytes=used_bytes,
            inode_count=inode_count,
            index_bytes=index_bytes,
        )

    async def _snapshot_and_upload_index(
        self,
        *,
        meta_dir: str,
        tmp_prefix: str,
        flush_live: bool = True,
        counter_bump: Optional[int] = None,
    ) -> "tuple[File, int]":
        """Snapshot the metadata index on disk at ``meta_dir`` and upload it.

        When ``flush_live=True`` (default), assumes a running daemon and
        flushes in-memory state before snapshotting: ``SAVE`` for Redis,
        WAL checkpoint for SQLite. Does not unmount and does not drain
        the writeback queue — chunks already written but not yet uploaded
        may still be local-only when the snapshot is published.

        When ``flush_live=False``, treats the on-disk index as authoritative
        (no daemon expected to be running). Used by :meth:`fork` when the
        Volume isn't mounted.

        For SQLite the snapshot is produced via SQLite's online backup API
        (which works equally well on a cold file). For Redis the snapshot is
        a plain ``shutil.copyfile`` of ``dump.rdb``.

        When ``counter_bump`` is provided, the snapshot's chunk/inode/session
        allocators are advanced by that offset before upload — used by
        :meth:`fork` to disjoint the parent's and child's allocator spaces.
        """
        engine = self._engine()
        src = Path(meta_dir) / _index_filename(engine)

        if flush_live:
            # For Redis the dump.rdb file is only materialized by SAVE
            # (server is started with --save ""), so we can't check
            # existence before the flush. WAL checkpoint on SQLite is a
            # no-op against a missing file but the existence check below
            # will catch that case explicitly.
            await asyncio.to_thread(os.sync)
            if engine == "redis":
                await asyncio.to_thread(_redis_save)
            else:
                if not src.exists():
                    raise RuntimeError(f"Cannot snapshot: no live index at {src}. Call mount() first.")
                await asyncio.to_thread(_wal_checkpoint, str(src))

        if not src.exists():
            raise RuntimeError(f"Cannot snapshot: no index at {src}.")

        if engine == "redis":

            def _snapshot() -> str:
                import shutil

                tmp = tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=".rdb", delete=False)
                tmp.close()
                shutil.copyfile(str(src), tmp.name)
                return tmp.name
        else:

            def _snapshot() -> str:
                tmp = tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=".db", delete=False)
                tmp.close()
                with sqlite3.connect(str(src)) as live, sqlite3.connect(tmp.name) as dst:
                    live.backup(dst)
                return tmp.name

        t_snap = time.monotonic()
        snapshot_path = await asyncio.to_thread(_snapshot)
        snap_bytes = os.path.getsize(snapshot_path)
        _logger.info(
            "[Volume.fork] snapshot prepared in %.2fs (%.1f MB, engine=%s)",
            time.monotonic() - t_snap,
            snap_bytes / 1_000_000,
            engine,
        )
        if counter_bump is not None:
            t_bump = time.monotonic()
            applied = await asyncio.to_thread(_disjoint_fork_counters, snapshot_path, engine, counter_bump)
            _logger.info(
                "[Volume.fork] counter bump applied in %.2fs (offset=%d)",
                time.monotonic() - t_bump,
                applied,
            )
            if applied != counter_bump:
                _logger.info(
                    "[Volume.fork] counter offset clamped: desired=%d applied=%d (uint64 headroom)",
                    counter_bump,
                    applied,
                )
        # Upload directly through the underlying fsspec filesystem (no signed
        # URLs — those are several times slower than native S3/GCS PUT).
        # We deliberately do NOT pass a hash_method: the streaming-hash path
        # in File.from_local wraps aiofiles in AsyncHashingReader which
        # iterates line-by-line on the binary input, producing thousands of
        # tiny S3 writes (observed ~234 KB/s on 100 MB). Volume indexes are
        # content-unique per fork (counter bump) so cache-key fidelity is
        # moot here — drop the hash and let fsspec.put do a proper multipart.
        try:
            index_bytes = os.path.getsize(snapshot_path)
            ctx = internal_ctx()
            dst_path = ctx.raw_data.get_random_remote_path(file_name=_index_filename(engine))
            t_up = time.monotonic()
            new_file: File = await File.from_local(
                snapshot_path,
                remote_destination=dst_path,
            )
            _logger.info(
                "[Volume.fork] uploaded in %.2fs (%.1f MB → %s)",
                time.monotonic() - t_up,
                index_bytes / 1_000_000,
                dst_path,
            )
            return new_file, index_bytes
        finally:
            try:
                os.unlink(snapshot_path)
            except OSError:
                pass

    async def fork(
        self,
        name: str,
        *,
        meta_dir: str = _DEFAULT_META_DIR,
    ) -> "Volume":
        """Snapshot the current metadata index and return a new ``Volume``
        that points at the snapshot.

        Both the original and the fork reference the same bucket. To keep
        their writes from clobbering each other, the fork's chunk-slice /
        inode / session counters are advanced by a random 56-bit offset
        before publication. JuiceFS chunk object keys embed the slice ID
        (``<bucket>/<vol>/chunks/<sliceid/1000>/<sliceid/1000>/<sliceid>_<index>_<size>``),
        so disjoint counter spaces yield disjoint object keys; without this,
        parent and fork would race to allocate the same slice IDs and one
        side's writes would silently overwrite the other's.

        Works whether or not ``self`` is currently mounted:

        * **Live** (mounted): flushes in-memory state (``SAVE`` for Redis,
          WAL checkpoint for SQLite) and snapshots the live on-disk index,
          bumps counters on the snapshot, and uploads it.
        * **Cold** (not mounted): downloads ``self.index`` to a tempdir,
          bumps counters in place, and uploads. Stats are inherited from
          ``self`` since no writes can have occurred.

        Note: cold-fork still avoids copying the data chunks (which dominate
        bytes), but it does pull the metadata file through the pod — the
        previous ``File.copy_to`` server-side path could not mutate counters
        and was unsafe for the chunk-key reasons above.
        """
        engine = self._engine()
        # "Live" iff mount() registered this meta_dir and commit() hasn't
        # cleared it. This is independent of file-on-disk state — for SQLite
        # the index.db lingers after commit(), so a file-existence heuristic
        # would falsely mark a committed-then-not-remounted volume as live.
        live = meta_dir in type(self)._live_meta
        offset = _random_fork_offset()
        _logger.info("[Volume.fork] disjoint counter offset=%d (live=%s)", offset, live)

        if live:
            new_index, index_bytes = await self._snapshot_and_upload_index(
                meta_dir=meta_dir,
                tmp_prefix="vol-fork-",
                flush_live=True,
                counter_bump=offset,
            )
            used_bytes, inode_count = await _query_volume_stats(meta_dir, engine)
        else:
            if self.index is None:
                raise RuntimeError("Cannot fork: Volume is not mounted and has no index to fork from.")
            # Cold fork: download the index into a private meta_dir, then run
            # the standard snapshot-and-upload path with flush_live=False so
            # the cold file is treated as authoritative. The same path also
            # applies the counter bump.
            cold_meta = Path(tempfile.mkdtemp(prefix="vol-cold-fork-"))
            try:
                index_path = cold_meta / _index_filename(engine)
                _logger.info("[Volume.fork] cold fork: downloading %s", self.index.path)
                t_dl = time.monotonic()
                await self.index.download(str(index_path))
                _logger.info(
                    "[Volume.fork] cold fork: downloaded in %.2fs (%.1f MB)",
                    time.monotonic() - t_dl,
                    os.path.getsize(index_path) / 1_000_000,
                )
                new_index, index_bytes = await self._snapshot_and_upload_index(
                    meta_dir=str(cold_meta),
                    tmp_prefix="vol-fork-",
                    flush_live=False,
                    counter_bump=offset,
                )
            finally:
                import shutil

                shutil.rmtree(cold_meta, ignore_errors=True)
            # No daemon, no writes since self.index was published — stats unchanged.
            used_bytes, inode_count = self.used_bytes, self.inode_count

        return Volume(
            name=name,
            bucket=self.bucket,
            storage=self.storage,
            index=new_index,
            parent=self.index,
            metadata_engine=self.metadata_engine,
            used_bytes=used_bytes,
            inode_count=inode_count,
            index_bytes=index_bytes,
        )

    async def commit_inplace(
        self,
        *,
        meta_dir: str = _DEFAULT_META_DIR,
    ) -> "Volume":
        """Snapshot and publish the live index without unmounting the volume.

        Like :meth:`commit`, but keeps the FUSE mount running and does NOT
        drain the writeback queue. Intended for periodic checkpoints of a
        long-running task so a hard pod kill loses at most the time since
        the last call.

        Returns a new ``Volume`` with the same ``name`` / ``bucket`` /
        ``storage`` / ``metadata_engine`` as ``self``, a fresh ``index``
        pointing at the uploaded snapshot, and ``parent`` linking to the
        previous index.

        Caveats:
          * The writeback queue is not flushed. Chunks written shortly
            before this call may still be local-only when the snapshot
            uploads; if the pod dies before they finish, the next mount
            of this snapshot will see file-not-found for those chunks.
          * Future writes after this call land in the same FUSE mount and
            will appear in subsequent snapshots — there is no semantic
            "branching" the way :meth:`fork` implies.
        """
        new_index, index_bytes = await self._snapshot_and_upload_index(meta_dir=meta_dir, tmp_prefix="vol-checkpoint-")
        return Volume(
            name=self.name,
            bucket=self.bucket,
            storage=self.storage,
            index=new_index,
            parent=self.index,
            metadata_engine=self.metadata_engine,
            index_bytes=index_bytes,
        )

    async def migrate_metadata_engine(
        self,
        new_engine: str,
        *,
        meta_dir: str = _DEFAULT_META_DIR,
        new_meta_dir: Optional[str] = None,
    ) -> "Volume":
        """Re-host this Volume's metadata on ``new_engine`` without copying
        data chunks.

        ``new_engine`` must be ``"sqlite"`` or ``"redis"`` and differ from the
        current engine. Implemented as ``juicefs dump | juicefs load``: the
        full namespace is exported to JSON, then re-imported into a fresh
        meta engine pointing at the *same* bucket. Chunks are addressed by
        slice IDs, which are preserved across dump/load, so no chunk traffic
        is required.

        The returned Volume has a fresh ``index`` (snapshot of the loaded
        meta engine), the original Volume's index as ``parent``, the same
        ``bucket`` / ``storage`` / ``name``, and the new ``metadata_engine``.

        Intent is *migration*, not fork: the old engine is meant to be
        retired. As a defense against accidental concurrent use, the loaded
        engine's chunk-slice / inode / session counters are advanced by a
        random offset (same mechanism as :meth:`fork`) so that even if the
        old engine is still mounted somewhere, its writes can't collide with
        the migrated engine's writes in shared object-store keys.

        Does not require a FUSE mount on either side. Safe to call from any
        task pod that has the ``juicefs`` binary (Redis tooling is only
        needed if one of the engines is ``"redis"``).
        """
        src_engine = self._engine()
        if new_engine == src_engine:
            raise ValueError(f"Source and destination engines are both {new_engine!r}; nothing to migrate.")
        if new_engine not in ("sqlite", "redis"):
            raise ValueError(f"Unsupported metadata_engine={new_engine!r}; expected 'sqlite' or 'redis'.")
        if self.index is None:
            raise RuntimeError("Cannot migrate metadata: this Volume has no index. Was it ever committed?")

        src_meta = Path(meta_dir) / "src"
        src_meta.mkdir(parents=True, exist_ok=True)
        src_index = src_meta / _index_filename(src_engine)
        _logger.info("[Volume.migrate_metadata_engine] downloading source index %s", self.index.path)
        await self.index.download(str(src_index))

        # `new_meta_dir` defaults to a sibling subdir under the same emptyDir-backed
        # parent as `meta_dir`. Keeping both under the same writable mount avoids
        # read-only rootfs surprises (e.g. /var/lib is RO in some base images).
        new_meta = Path(new_meta_dir or str(Path(meta_dir) / "new"))
        new_meta.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(prefix="vol-migrate-", suffix=".json", delete=False) as f:
            dump_path = f.name

        src_redis: Optional[subprocess.Popen] = None
        new_redis: Optional[subprocess.Popen] = None
        cleanup_paths: list[str] = [dump_path]
        try:
            if src_engine == "redis":
                src_redis = await _start_redis(str(src_meta))

            src_meta_url = _meta_url(str(src_meta), src_engine)
            _logger.info("[Volume.migrate_metadata_engine] dump %s -> %s", src_meta_url, dump_path)
            await asyncio.to_thread(_run_check, [_CLIENT_BINARY, "dump", src_meta_url, dump_path])

            # Stop src redis before starting dst redis — both want :6379.
            if src_redis is not None:
                _stop_redis(src_redis, timeout=10.0)
                src_redis = None

            if new_engine == "redis":
                new_redis = await _start_redis(str(new_meta))

            new_meta_url = _meta_url(str(new_meta), new_engine)
            _logger.info("[Volume.migrate_metadata_engine] load %s <- %s", new_meta_url, dump_path)
            await asyncio.to_thread(_run_check, [_CLIENT_BINARY, "load", new_meta_url, dump_path])

            new_index_path = new_meta / _index_filename(new_engine)
            if new_engine == "redis":
                # Flush the loaded namespace to dump.rdb and stop the daemon
                # so we get a stable file to snapshot.
                await asyncio.to_thread(_redis_save)
                _stop_redis(new_redis, timeout=10.0)
                new_redis = None
            else:
                await asyncio.to_thread(_wal_checkpoint, str(new_index_path))

            if not new_index_path.exists():
                raise RuntimeError(f"juicefs load did not produce {new_index_path} — migration aborted.")

            # Disjoint the migrated engine's allocator space from the source
            # engine's. See class:`Volume.fork` for the rationale: even though
            # migration intends the source to be retired, defense in depth.
            offset = _random_fork_offset()
            _logger.info("[Volume.migrate_metadata_engine] disjoint counter offset=%d", offset)
            await asyncio.to_thread(_disjoint_fork_counters, str(new_index_path), new_engine, offset)

            def _snapshot() -> str:
                import shutil

                suffix = new_index_path.suffix or ".idx"
                tmp = tempfile.NamedTemporaryFile(prefix="vol-migrate-", suffix=suffix, delete=False)
                tmp.close()
                shutil.copyfile(str(new_index_path), tmp.name)
                return tmp.name

            snapshot_path = await asyncio.to_thread(_snapshot)
            cleanup_paths.append(snapshot_path)
            ctx = internal_ctx()
            index_bytes = os.path.getsize(snapshot_path)
            # No hash_method — see fork()'s upload site for why.
            new_file: File = await File.from_local(
                snapshot_path,
                remote_destination=ctx.raw_data.get_random_remote_path(file_name=_index_filename(new_engine)),
            )
            return Volume(
                name=self.name,
                bucket=self.bucket,
                storage=self.storage,
                index=new_file,
                parent=self.index,
                metadata_engine=new_engine,
                index_bytes=index_bytes,
            )
        finally:
            if src_redis is not None:
                _stop_redis(src_redis, timeout=5.0)
            if new_redis is not None:
                _stop_redis(new_redis, timeout=5.0)
            for path in cleanup_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _client_bucket_uri(bucket: str, storage: str) -> str:
    """Translate a clean storage URI (e.g. ``s3://my-bucket/prefix``) into the
    form the volume client expects (``https://s3.{region}.amazonaws.com/my-bucket/prefix``
    for S3). Pass-through for already-translated or other-scheme URIs.
    """
    if storage == "s3" and bucket.startswith("s3://"):
        rest = bucket[len("s3://") :]
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        return f"https://s3.{region}.amazonaws.com/{rest}"
    return bucket


def _default_bucket() -> str:
    """Derive the default volume bucket from the active task context.

    Constructs ``{stable-prefix}/{project}/{domain}/volumes`` where the
    stable-prefix is whatever Flyte's ``raw_data_path`` is rooted at — i.e.
    everything *before* the project segment, so any org-level subprefix
    (e.g. ``s3://foo-bucket/xy/``) is preserved.
    """
    ctx = internal_ctx()
    tctx = ctx.data.task_context if ctx.data else None
    if tctx is None:
        raise RuntimeError(
            "Volume.empty() requires either an explicit `bucket=` argument or "
            "an active task context (so the bucket can be derived from raw_data_path)."
        )

    project = tctx.action.project
    domain = tctx.action.domain
    raw = tctx.raw_data_path.path
    if not project or not domain:
        raise RuntimeError(
            f"Cannot derive default bucket: task context is missing project/domain "
            f"(project={project!r}, domain={domain!r}). Pass `bucket=` explicitly."
        )

    marker = f"/{project}/{domain}/"
    idx = raw.find(marker)
    if idx < 0:
        raise RuntimeError(
            f"Cannot derive default bucket: raw_data_path={raw!r} doesn't contain "
            f"/{project}/{domain}/. Pass `bucket=` explicitly."
        )
    base = raw[:idx]
    return f"{base}/{project}/{domain}/volumes"


async def _query_volume_stats(meta_dir: str, engine: str) -> tuple[Optional[int], Optional[int]]:
    """Best-effort: return ``(used_bytes, inode_count)`` for the volume backed
    by ``meta_dir`` / ``engine``. Shells out to ``juicefs status`` and parses
    the JSON. Returns ``(None, None)`` on any failure so commit/fork are never
    blocked by stats collection.

    Must be called while the metadata engine is reachable — for Redis, before
    the in-process redis-server is shut down; for SQLite, any time after the
    index file is on disk.
    """
    import json

    meta_url = _meta_url(meta_dir, engine)
    try:
        r = await asyncio.to_thread(
            subprocess.run,
            [_CLIENT_BINARY, "status", meta_url],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            _logger.warning("[Volume.stats] juicefs status rc=%d: %s", r.returncode, r.stderr[:200])
            return None, None
        data = json.loads(r.stdout)
        stat = data.get("Statistic") or {}
        used_bytes: Optional[int] = None
        inode_count: Optional[int] = None
        raw_used = stat.get("UsedSpace")
        if isinstance(raw_used, (int, float)):
            used_bytes = int(raw_used)
        raw_inodes = stat.get("UsedInodes")
        if isinstance(raw_inodes, (int, float)):
            inode_count = int(raw_inodes)
        return used_bytes, inode_count
    except Exception as e:
        _logger.warning("[Volume.stats] failed to query stats: %s", e)
        return None, None


def _run_check(cmd: list) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _is_fuse_mount(path: str) -> bool:
    """True if *path* is a FUSE mount (not just any mountpoint).

    The emptyDir Kubernetes injects at the mount path is itself a bind-mount,
    so a plain "is this a mountpoint?" check would return True before the
    volume client has had a chance to overlay its FUSE filesystem.
    """
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                # Format: id parent maj:min root mountpoint opts ... - fstype source super_opts
                parts = line.split()
                if len(parts) < 5 or parts[4] != path:
                    continue
                try:
                    sep = parts.index("-")
                except ValueError:
                    continue
                if len(parts) > sep + 1 and parts[sep + 1].startswith("fuse"):
                    return True
    except OSError:
        pass
    return False


async def _start_redis(meta_dir: str, timeout: float = 30.0) -> subprocess.Popen:
    """Spawn an in-process ``redis-server`` rooted at ``meta_dir``.

    Redis is configured with auto-save disabled (``--save ""``) and AOF off;
    persistence is driven explicitly by ``commit()`` / ``fork()`` via
    ``redis-cli SAVE``. If ``dump.rdb`` already exists in ``meta_dir``,
    redis-server loads it during startup.
    """
    cmd = [
        "redis-server",
        "--port",
        str(_REDIS_PORT),
        "--bind",
        "127.0.0.1",
        "--save",
        "",
        "--appendonly",
        "no",
        "--dir",
        meta_dir,
        "--dbfilename",
        _REDIS_INDEX_FILENAME,
        "--daemonize",
        "no",
    ]
    _logger.info("[Volume.mount] starting redis-server in %s", meta_dir)
    proc = subprocess.Popen(  # noqa: ASYNC220 - long-lived daemon held in a class-level dict
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"redis-server exited prematurely (rc={proc.returncode}): {out}")
        r = await asyncio.to_thread(
            subprocess.run,
            ["redis-cli", "-p", str(_REDIS_PORT), "ping"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and "PONG" in r.stdout:
            _logger.info("[Volume.mount] redis-server ready")
            return proc
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"redis-server did not become ready within {timeout}s")
        await asyncio.sleep(0.2)


def _redis_save() -> None:
    """Trigger a synchronous RDB save."""
    _run_check(["redis-cli", "-p", str(_REDIS_PORT), "SAVE"])


def _stop_redis(proc: Optional[subprocess.Popen], timeout: float) -> None:
    """Shut down a redis-server cleanly. Caller must have already saved if
    they want the in-memory state persisted.
    """
    if proc is None:
        return
    # SHUTDOWN NOSAVE — we drove persistence ourselves via SAVE.
    subprocess.run(
        ["redis-cli", "-p", str(_REDIS_PORT), "SHUTDOWN", "NOSAVE"],
        capture_output=True,
        check=False,
    )
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _logger.warning("[Volume.commit] redis-server didn't exit in %ss; killing", timeout)
        proc.kill()
        proc.wait(5)


def _wal_checkpoint(db_path: str) -> None:
    """Force a TRUNCATE-mode WAL checkpoint so the on-disk .db contains all
    committed transactions. No-op if the DB isn't in WAL mode.
    """
    try:
        conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        pass


# Counters JuiceFS uses to allocate IDs. Bumped at fork time so parent and
# fork allocate from disjoint ranges (chunk-slice IDs end up in distinct
# object-store keys; inode/session IDs stay distinct just for hygiene).
# SQLite stores them in ``jfs_counter(name, value)`` with camelCase names;
# Redis stores them as top-level keys with lowercase names.
_FORK_COUNTER_NAMES_SQLITE: tuple = ("nextChunk", "nextInode", "nextSession")
_FORK_COUNTER_NAMES_REDIS: tuple = ("nextchunk", "nextinode", "nextsession")

# JuiceFS counter ceiling. SQLite ``INTEGER`` is signed 64-bit (max 2**63 - 1)
# and Redis ``INCRBY`` is also signed 64-bit, so the effective ceiling is
# 2**63 - 1, not 2**64 - 1.
_COUNTER_MAX = (1 << 63) - 1
# Minimum offset we'll apply to a counter at fork time. Comfortably exceeds
# JuiceFS's per-batch allocation (4096 chunk slices) and a generous per-task
# write rate, so even a forced-small offset still leaves enough room for one
# more task to allocate without colliding with sibling forks.
_MIN_FORK_OFFSET = 1 << 32


def _random_fork_offset() -> int:
    """Return a random offset in ``[2³², 2³² + 2⁵⁶)`` — the *desired* fork
    offset, before any counter-headroom clamping.

    The lower bound (2³²) ensures every fork's allocator space is at least
    ~4B IDs ahead of the parent — JuiceFS allocates chunk-slice IDs in
    batches of 4096, so this is comfortably more than any realistic burst.
    The 56-bit upper bound keeps successive forks well-clear of uint64
    overflow under typical use; for deep fork chains
    :func:`_safe_fork_offset` shrinks the actual applied offset based on the
    counter's remaining headroom.
    Birthday-collision probability for 10⁶ siblings is on the order of 10⁻⁵.
    """
    return (1 << 32) + secrets.randbits(56)


def _safe_fork_offset(current_max: int, desired: int) -> int:
    """Clamp a desired fork offset so that ``current_max + offset`` stays
    well below :data:`_COUNTER_MAX` (signed int64).

    The clamp targets ``headroom // 2`` so each successive fork still has
    half the remaining ID space available — protecting future forks rather
    than starving them. Raises if even the minimum safe offset
    (:data:`_MIN_FORK_OFFSET`) would not leave room for one further fork,
    which means the Volume's allocator space is exhausted and the volume
    must be re-created.
    """
    if desired <= 0:
        raise ValueError(f"desired offset must be positive, got {desired}")
    headroom = _COUNTER_MAX - current_max
    if headroom < _MIN_FORK_OFFSET * 2:
        raise RuntimeError(
            f"JuiceFS counter is too close to int64 max (current_max={current_max}); "
            f"this Volume's fork chain has exhausted its ID space — re-create the volume "
            f"to reset the allocator."
        )
    return min(desired, headroom // 2)


def _disjoint_counters_sqlite(db_path: str, offset: int) -> int:
    """Advance JuiceFS allocator counters in a SQLite metadata index by an
    offset clamped to the uint64 headroom. Returns the actual offset applied.

    Raises if the schema or counters are missing — silently failing here
    would let collisions through — or if the allocator space is exhausted.
    """
    if offset <= 0:
        raise ValueError(f"offset must be positive, got {offset}")
    with sqlite3.connect(db_path) as conn:
        existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "jfs_counter" not in existing:
            raise RuntimeError(f"{db_path} is not a JuiceFS SQLite index (no jfs_counter table)")
        cur = conn.cursor()
        current_max = 0
        for name in _FORK_COUNTER_NAMES_SQLITE:
            row = cur.execute("SELECT value FROM jfs_counter WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise RuntimeError(f"missing counter {name!r} in {db_path}")
            if row[0] > current_max:
                current_max = row[0]
        applied = _safe_fork_offset(current_max, offset)
        for name in _FORK_COUNTER_NAMES_SQLITE:
            cur.execute(
                "UPDATE jfs_counter SET value = value + ? WHERE name = ?",
                (applied, name),
            )
        conn.commit()
    return applied


def _disjoint_counters_redis(rdb_path: str, offset: int, timeout: float = 30.0) -> int:
    """Advance JuiceFS allocator counters in a Redis RDB file by an offset
    clamped to the uint64 headroom. Returns the actual offset applied.

    Spawns an ephemeral ``redis-server`` on an OS-assigned port that loads
    the RDB, applies ``INCRBY`` to each counter, ``SAVE``s, and shuts down.
    The file is rewritten in place.
    """
    if offset <= 0:
        raise ValueError(f"offset must be positive, got {offset}")
    rdb = Path(rdb_path)
    if not rdb.exists():
        raise RuntimeError(f"{rdb_path} does not exist")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cmd = [
        "redis-server",
        "--port",
        str(port),
        "--bind",
        "127.0.0.1",
        "--save",
        "",
        "--appendonly",
        "no",
        "--dir",
        str(rdb.parent),
        "--dbfilename",
        rdb.name,
        "--daemonize",
        "no",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        # Wait for PONG.
        import time

        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                raise RuntimeError(f"ephemeral redis-server exited prematurely (rc={proc.returncode}): {out}")
            r = subprocess.run(
                ["redis-cli", "-p", str(port), "ping"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode == 0 and "PONG" in r.stdout:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ephemeral redis-server did not become ready within {timeout}s")
            time.sleep(0.1)

        # Verify the RDB was actually a JuiceFS index — counters must exist.
        # Also collect current max so we can clamp the offset to remaining headroom.
        current_max = 0
        for name in _FORK_COUNTER_NAMES_REDIS:
            r = subprocess.run(
                ["redis-cli", "-p", str(port), "GET", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode != 0 or r.stdout.strip() in ("", "(nil)"):
                raise RuntimeError(f"missing counter {name!r} in {rdb_path}")
            try:
                v = int(r.stdout.strip())
            except ValueError as e:
                raise RuntimeError(f"counter {name!r} in {rdb_path} is non-integer: {r.stdout!r}") from e
            if v > current_max:
                current_max = v
        applied = _safe_fork_offset(current_max, offset)
        for name in _FORK_COUNTER_NAMES_REDIS:
            _run_check(["redis-cli", "-p", str(port), "INCRBY", name, str(applied)])

        _run_check(["redis-cli", "-p", str(port), "SAVE"])
    finally:
        subprocess.run(
            ["redis-cli", "-p", str(port), "SHUTDOWN", "NOSAVE"],
            capture_output=True,
            check=False,
        )
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)
    return applied


def _disjoint_fork_counters(index_path: str, engine: str, offset: int) -> int:
    """Dispatch ``_disjoint_counters_{sqlite,redis}`` by ``engine``. Returns
    the offset actually applied (may be smaller than ``offset`` when the
    counter space is nearly exhausted).
    """
    if engine == "redis":
        return _disjoint_counters_redis(index_path, offset)
    return _disjoint_counters_sqlite(index_path, offset)


def volume_image(
    base: Optional[Image] = None,
    *,
    version: str = _CLIENT_VERSION,
    architecture: str = "amd64",
) -> Image:
    """Layer the volume client and the ``fuse`` userspace tools onto ``base``
    (a :class:`flyte.Image`). Returns a new :class:`flyte.Image`.

    The primary container needs the volume client binary and the
    ``fusermount`` helper because :class:`Volume` mounts in-process rather
    than via a sidecar.
    """
    if base is None:
        base = Image.from_debian_base(install_flyte=False)

    url = (
        f"https://github.com/juicedata/juicefs/releases/download/"
        f"v{version}/juicefs-{version}-linux-{architecture}.tar.gz"
    )
    install_cmd = (
        f"set -e; "
        f"curl -fsSL -o /tmp/client.tar.gz {url}; "
        f"tar -xzf /tmp/client.tar.gz -C /tmp {_CLIENT_BINARY}; "
        f"install -m 0755 /tmp/{_CLIENT_BINARY} /usr/local/bin/{_CLIENT_BINARY}; "
        f"rm /tmp/client.tar.gz /tmp/{_CLIENT_BINARY}; "
        # fusermount looks up /etc/mtab on umount; provide a symlink to
        # /proc/mounts so unmounts don't ENOENT.
        f"ln -sf /proc/mounts /etc/mtab; "
        f"{_CLIENT_BINARY} version"
    )
    return base.with_apt_packages("ca-certificates", "curl", "fuse", "redis-server", "redis-tools").with_commands(
        [install_cmd]
    )


def volume_pod_template(
    *,
    mount_path: str = _DEFAULT_MOUNT_PATH,
    meta_dir: str = _DEFAULT_META_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    cache_size_gb: int = 50,
    primary_container_name: str = "primary",
) -> PodTemplate:
    """Build a :class:`flyte.PodTemplate` that lets the primary container
    mount a :class:`Volume` in-process.

    The returned template:

    * adds ``emptyDir`` volumes for the metadata directory, the chunk cache
      (sized via ``cache_size_gb``), and the mount point;
    * adds a ``hostPath`` volume for ``/dev/fuse``;
    * makes the primary container privileged with ``CAP_SYS_ADMIN`` so it
      can perform the FUSE mount itself.

    ``cache_size_gb`` caps the on-node ephemeral storage used for the
    write-back cache. Size it for your task's working set — writes that
    overflow the cache will block until existing chunks finish uploading.

    The volume's identity (bucket, name, storage backend) is *not* baked into
    the template — it is carried by the ``Volume`` input at runtime.
    """
    from kubernetes.client import (
        V1Capabilities,
        V1Container,
        V1EmptyDirVolumeSource,
        V1HostPathVolumeSource,
        V1PodSpec,
        V1SecurityContext,
        V1Volume,
        V1VolumeMount,
    )

    primary = V1Container(
        name=primary_container_name,
        security_context=V1SecurityContext(
            privileged=True,
            capabilities=V1Capabilities(add=["SYS_ADMIN"]),
        ),
        volume_mounts=[
            V1VolumeMount(name="vol-meta", mount_path=meta_dir),
            V1VolumeMount(name="vol-cache", mount_path=cache_dir),
            V1VolumeMount(name="vol-workspace", mount_path=mount_path),
            V1VolumeMount(name="fuse-device", mount_path="/dev/fuse"),
        ],
    )

    pod_spec = V1PodSpec(
        containers=[primary],
        volumes=[
            V1Volume(name="vol-meta", empty_dir=V1EmptyDirVolumeSource()),
            V1Volume(
                name="vol-cache",
                empty_dir=V1EmptyDirVolumeSource(size_limit=f"{cache_size_gb}Gi"),
            ),
            V1Volume(name="vol-workspace", empty_dir=V1EmptyDirVolumeSource()),
            V1Volume(
                name="fuse-device",
                host_path=V1HostPathVolumeSource(path="/dev/fuse", type="CharDevice"),
            ),
        ],
    )

    return PodTemplate(pod_spec=pod_spec, primary_container_name=primary_container_name)


__all__ = ["Volume", "volume_image", "volume_pod_template"]
