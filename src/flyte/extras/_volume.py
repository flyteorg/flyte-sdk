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
from flyte.extras._volume_backend import _MetadataEngineState, _MountConfig, _VolumeBackend
from flyte.extras._volume_juicefs import _CLIENT_BINARY, _CLIENT_VERSION, JuiceFSVolumeBackend
from flyte.extras._volume_local import local_commit, local_commit_inplace, local_fork, local_mount
from flyte.io._file import File

_DEFAULT_META_DIR = "/var/lib/flyte-volume"
_DEFAULT_CACHE_DIR = "/var/cache/flyte-volume"
_DEFAULT_MOUNT_PATH = "/workspace"

_VOLUME_BACKENDS: dict[str, _VolumeBackend] = {"juicefs": JuiceFSVolumeBackend()}


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
    # Runtime backend. ``None``/missing resolves to ``"juicefs"`` for
    # backward compatibility with Volumes serialized before this field existed.
    # ``"local"`` selects the tar-archive backend (no FUSE, no daemon, no
    # privileged container required); ``"juicefs"`` uses the JuiceFS client.
    volume_backend: Optional[str] = None
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
    # Original mount options, keyed by mount path, so live fork can remount
    # after a flush-only unmount without changing caller-selected tuning.
    _live_mounts: ClassVar[Dict[str, _MountConfig]] = {}
    # Track in-process redis-server daemons, keyed by meta_dir.
    _live_redis: ClassVar[Dict[str, _MetadataEngineState]] = {}
    # meta_dirs that currently have a live engine + daemon. Used by fork()
    # to decide whether to use the live flush-and-upload path or the cold
    # File.copy_to path. Populated by mount(), cleared by commit().
    _live_meta: ClassVar[Set[str]] = set()

    def _engine(self) -> str:
        return self.metadata_engine or "sqlite"

    def _backend(self) -> _VolumeBackend:
        return _VOLUME_BACKENDS["juicefs"]

    @classmethod
    def empty(
        cls,
        name: str,
        bucket: Optional[str] = None,
        *,
        storage: str = "s3",
        metadata_engine: Optional[str] = None,
        volume_backend: str = "juicefs",
    ) -> "Volume":
        """Declare a brand-new volume. The first ``mount()`` call will
        bootstrap the namespace (the underlying client refuses to format
        over a non-empty bucket prefix).

        If ``bucket`` is omitted, it is derived from the currently active
        task context as ``{raw_data_root}/{project}/{domain}/volumes`` —
        following Flyte's own layout for offloaded data. Must be called
        from inside a task in that case.

        ``volume_backend`` selects the runtime backend:

        * ``"juicefs"`` (default) — JuiceFS-backed FUSE filesystem. Chunked
          object-store storage, copy-on-write forks, supports large volumes.
          Requires a privileged container with FUSE access.
        * ``"local"`` — a single tar.gz archive in object storage. No FUSE,
          no daemon, works in a regular non-privileged container. ``commit()``
          tars the whole tree; ``fork()`` is a server-side copy of the
          archive. Best for ≤ a few GB of data.

        ``metadata_engine`` controls the JuiceFS metadata backend (ignored for
        ``volume_backend="local"``). ``"redis"`` (default for JuiceFS) runs an
        in-process ``redis-server`` and persists the namespace as an RDB
        snapshot — faster than SQLite for metadata-heavy workloads and more
        compact on disk. ``"sqlite"`` keeps everything on local disk with no
        extra daemon. The choice is baked into the Volume and travels with
        it through lineage; subsequent mounts of the same Volume must use
        the same engine.
        """
        if bucket is None:
            bucket = _default_bucket()
        if metadata_engine is None and volume_backend == "juicefs":
            metadata_engine = "redis"
        return cls(
            name=name,
            bucket=bucket,
            storage=storage,
            index=None,
            parent=None,
            metadata_engine=metadata_engine,
            volume_backend=volume_backend,
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
        read_only: bool = False,
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
        if self.volume_backend == "local":
            await local_mount(self, mount_path=mount_path)
            return

        engine = self._engine()
        backend = self._backend()
        meta = Path(meta_dir)
        meta.mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(mount_path).mkdir(parents=True, exist_ok=True)

        index_path = meta / backend.index_filename(engine)
        if self.index is not None:
            _logger.info("[Volume.mount] downloading index %s", self.index.path)
            await self.index.download(str(index_path))
        elif index_path.exists():
            # Stale index from an earlier mount in the same pod (shouldn't
            # happen in practice; tasks get fresh pods).
            index_path.unlink()

        redis_state: Optional[_MetadataEngineState] = None
        try:
            redis_port: Optional[int] = None
            redis_state = await backend.start_metadata_engine(meta_dir, engine)
            if redis_state is not None:
                redis_port = redis_state.port
                type(self)._live_redis[meta_dir] = redis_state

            meta_url = backend.meta_url(meta_dir, engine, redis_port=redis_port)
            client_bucket = _client_bucket_uri(self.bucket, self.storage)
            if self.index is None:
                _logger.info(
                    "[Volume.mount] formatting fresh volume name=%s bucket=%s storage=%s",
                    self.name,
                    client_bucket,
                    self.storage,
                )
                await asyncio.to_thread(
                    backend.format,
                    storage=self.storage,
                    bucket=client_bucket,
                    meta_url=meta_url,
                    name=self.name,
                )

            mount_config = _MountConfig(
                meta_dir=meta_dir,
                redis_port=redis_port,
                cache_dir=cache_dir,
                writeback=writeback,
                upload_delay=upload_delay,
                max_uploads=max_uploads,
                attr_cache=attr_cache,
                entry_cache=entry_cache,
                dir_entry_cache=dir_entry_cache,
                read_only=read_only,
            )
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
            await self._start_mount(mount_path=mount_path, config=mount_config, timeout=timeout)
            type(self)._live_meta.add(meta_dir)
            _logger.info("[Volume.mount] %s is now a FUSE mount", mount_path)
        except Exception:
            type(self)._live_meta.discard(meta_dir)
            type(self)._live_procs.pop(mount_path, None)
            type(self)._live_mounts.pop(mount_path, None)
            if engine == "redis":
                state = type(self)._live_redis.pop(meta_dir, None) or redis_state
                await asyncio.to_thread(backend.stop_metadata_engine, state, timeout)
            raise

    async def _start_mount(self, *, mount_path: str, config: _MountConfig, timeout: float) -> None:
        """Start the volume client against an already-prepared metadata engine."""
        engine = self._engine()
        backend = self._backend()
        mount_cmd = backend.mount_cmd(config=config, engine=engine, mount_path=mount_path)

        proc = subprocess.Popen(  # noqa: ASYNC220 - long-lived daemon held in a class-level dict
            mount_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        type(self)._live_procs[mount_path] = proc
        type(self)._live_mounts[mount_path] = config

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if proc.poll() is not None:
                type(self)._live_procs.pop(mount_path, None)
                type(self)._live_mounts.pop(mount_path, None)
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                raise RuntimeError(f"volume client exited prematurely (rc={proc.returncode}): {out}")
            if backend.is_mounted(mount_path):
                return
            if asyncio.get_event_loop().time() >= deadline:
                type(self)._live_procs.pop(mount_path, None)
                type(self)._live_mounts.pop(mount_path, None)
                proc.kill()
                await asyncio.to_thread(proc.wait, 5)
                raise TimeoutError(f"mount at {mount_path} did not become a FUSE mountpoint within {timeout}s")
            await asyncio.sleep(0.5)

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
        if self.volume_backend == "local":
            return await local_commit(self, mount_path=mount_path)

        engine = self._engine()
        backend = self._backend()
        # Flush page cache so any unfsync'd writes hit the filesystem first.
        await asyncio.to_thread(backend.sync_filesystem, mount_path)

        meta = Path(meta_dir)
        index_path = meta / backend.index_filename(engine)

        # Tell the client to flush and exit cleanly.
        _logger.info("[Volume.commit] unmounting %s", mount_path)
        await asyncio.to_thread(backend.unmount, mount_path)

        proc = type(self)._live_procs.pop(mount_path, None)
        type(self)._live_mounts.pop(mount_path, None)
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
            redis_state = type(self)._live_redis.pop(meta_dir, None)
            redis_port = redis_state.port if redis_state is not None else None
            await asyncio.to_thread(backend.save_metadata, engine, redis_port=redis_port)
            used_bytes, inode_count = await backend.query_stats(meta_dir, engine, redis_port=redis_port)
            await asyncio.to_thread(backend.stop_metadata_engine, redis_state, timeout)
        else:
            # Force WAL pages back into the main .db before uploading.
            await asyncio.to_thread(backend.checkpoint_metadata, str(index_path))
            used_bytes, inode_count = await backend.query_stats(meta_dir, engine)

        type(self)._live_meta.discard(meta_dir)
        ctx = internal_ctx()
        index_bytes = os.path.getsize(str(index_path))
        # No hash_method — see fork()'s upload site for why (streaming-hash
        # path produces thousands of tiny S3 writes on binary indexes).
        new_index: File = await File.from_local(
            str(index_path),
            remote_destination=ctx.raw_data.get_random_remote_path(file_name=backend.index_filename(engine)),
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
        backend = self._backend()
        src = Path(meta_dir) / backend.index_filename(engine)

        if flush_live:
            # For Redis the dump.rdb file is only materialized by SAVE
            # (server is started with --save ""), so we can't check
            # existence before the flush. WAL checkpoint on SQLite is a
            # no-op against a missing file but the existence check below
            # will catch that case explicitly.
            await asyncio.to_thread(backend.sync_filesystem, meta_dir)
            if engine == "redis":
                redis_state = type(self)._live_redis.get(meta_dir)
                redis_port = redis_state.port if redis_state is not None else None
                await asyncio.to_thread(backend.save_metadata, engine, redis_port=redis_port)
            else:
                if not src.exists():
                    raise RuntimeError(f"Cannot snapshot: no live index at {src}. Call mount() first.")
                await asyncio.to_thread(backend.checkpoint_metadata, str(src))

        if not src.exists():
            raise RuntimeError(f"Cannot snapshot: no index at {src}.")

        t_snap = time.monotonic()
        snapshot_path = await asyncio.to_thread(backend.snapshot_index, src, engine, tmp_prefix)
        snap_bytes = os.path.getsize(snapshot_path)
        _logger.info(
            "[Volume.fork] snapshot prepared in %.2fs (%.1f MB, engine=%s)",
            time.monotonic() - t_snap,
            snap_bytes / 1_000_000,
            engine,
        )
        if counter_bump is not None:
            t_bump = time.monotonic()
            applied = await asyncio.to_thread(backend.disjoint_fork_counters, snapshot_path, engine, counter_bump)
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
            dst_path = ctx.raw_data.get_random_remote_path(file_name=backend.index_filename(engine))
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
        mount_path: str = _DEFAULT_MOUNT_PATH,
        meta_dir: str = _DEFAULT_META_DIR,
        timeout: float = 60.0,
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

        For the ``local`` backend, fork is a server-side copy of the
        committed archive — no counter bumps are needed because every
        archive is independent.
        """
        if self.volume_backend == "local":
            return await local_fork(self, name)

        engine = self._engine()
        backend = self._backend()
        # "Live" iff mount() registered this meta_dir and commit() hasn't
        # cleared it. This is independent of file-on-disk state — for SQLite
        # the index.db lingers after commit(), so a file-existence heuristic
        # would falsely mark a committed-then-not-remounted volume as live.
        live = meta_dir in type(self)._live_meta
        offset = _random_fork_offset()
        _logger.info("[Volume.fork] disjoint counter offset=%d (live=%s)", offset, live)

        if live:
            mount_config = await self._flush_live_mount(mount_path=mount_path, meta_dir=meta_dir, timeout=timeout)
            try:
                new_index, index_bytes = await self._snapshot_and_upload_index(
                    meta_dir=meta_dir,
                    tmp_prefix="vol-fork-",
                    flush_live=True,
                    counter_bump=offset,
                )
                redis_state = type(self)._live_redis.get(meta_dir)
                redis_port = redis_state.port if redis_state is not None else None
                used_bytes, inode_count = await backend.query_stats(meta_dir, engine, redis_port=redis_port)
            finally:
                await self._start_mount(mount_path=mount_path, config=mount_config, timeout=timeout)
                type(self)._live_meta.add(meta_dir)
        else:
            if self.index is None:
                raise RuntimeError("Cannot fork: Volume is not mounted and has no index to fork from.")
            # Cold fork: download the index into a private meta_dir, then run
            # the standard snapshot-and-upload path with flush_live=False so
            # the cold file is treated as authoritative. The same path also
            # applies the counter bump.
            cold_meta = Path(tempfile.mkdtemp(prefix="vol-cold-fork-"))
            try:
                index_path = cold_meta / backend.index_filename(engine)
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

    async def _flush_live_mount(self, *, mount_path: str, meta_dir: str, timeout: float) -> _MountConfig:
        """Drain writeback chunks by unmounting with --flush."""
        backend = self._backend()
        config = type(self)._live_mounts.get(mount_path)
        if config is None or config.meta_dir != meta_dir:
            raise RuntimeError(f"Cannot flush live Volume: {mount_path} is not registered for meta_dir={meta_dir}.")

        _logger.info("[Volume.fork] flushing live mount %s before snapshot", mount_path)
        await asyncio.to_thread(backend.unmount, mount_path, flush=True)

        proc = type(self)._live_procs.pop(mount_path, None)
        type(self)._live_mounts.pop(mount_path, None)
        type(self)._live_meta.discard(meta_dir)
        if proc is not None:
            try:
                await asyncio.to_thread(proc.wait, timeout)
            except subprocess.TimeoutExpired:
                _logger.warning("[Volume.fork] client didn't exit in %ss after flush; killing", timeout)
                proc.kill()
                await asyncio.to_thread(proc.wait, 5)

        return config

    async def commit_inplace(
        self,
        *,
        mount_path: str = _DEFAULT_MOUNT_PATH,
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

        For the ``local`` backend this is equivalent to :meth:`commit` —
        there's no live daemon to keep running, so we just re-tar the
        mount point.
        """
        if self.volume_backend == "local":
            return await local_commit_inplace(self, mount_path=mount_path)

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
        backend = self._backend()
        src_engine = self._engine()
        if new_engine == src_engine:
            raise ValueError(f"Source and destination engines are both {new_engine!r}; nothing to migrate.")
        if new_engine not in ("sqlite", "redis"):
            raise ValueError(f"Unsupported metadata_engine={new_engine!r}; expected 'sqlite' or 'redis'.")
        if self.index is None:
            raise RuntimeError("Cannot migrate metadata: this Volume has no index. Was it ever committed?")

        src_meta = Path(meta_dir) / "src"
        src_meta.mkdir(parents=True, exist_ok=True)
        src_index = src_meta / backend.index_filename(src_engine)
        _logger.info("[Volume.migrate_metadata_engine] downloading source index %s", self.index.path)
        await self.index.download(str(src_index))

        # `new_meta_dir` defaults to a sibling subdir under the same emptyDir-backed
        # parent as `meta_dir`. Keeping both under the same writable mount avoids
        # read-only rootfs surprises (e.g. /var/lib is RO in some base images).
        new_meta = Path(new_meta_dir or str(Path(meta_dir) / "new"))
        new_meta.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(prefix="vol-migrate-", suffix=".json", delete=False) as f:
            dump_path = f.name

        src_redis: Optional[_MetadataEngineState] = None
        new_redis: Optional[_MetadataEngineState] = None
        cleanup_paths: list[str] = [dump_path]
        try:
            if src_engine == "redis":
                src_redis = await backend.start_metadata_engine(str(src_meta), src_engine)

            src_meta_url = backend.meta_url(str(src_meta), src_engine, redis_port=src_redis.port if src_redis else None)
            _logger.info("[Volume.migrate_metadata_engine] dump %s -> %s", src_meta_url, dump_path)
            await asyncio.to_thread(backend.dump_metadata, src_meta_url, dump_path)

            # Stop src redis before starting dst redis — both want :6379.
            if src_redis is not None:
                backend.stop_metadata_engine(src_redis, timeout=10.0)
                src_redis = None

            if new_engine == "redis":
                new_redis = await backend.start_metadata_engine(str(new_meta), new_engine)

            new_meta_url = backend.meta_url(str(new_meta), new_engine, redis_port=new_redis.port if new_redis else None)
            _logger.info("[Volume.migrate_metadata_engine] load %s <- %s", new_meta_url, dump_path)
            await asyncio.to_thread(backend.load_metadata, new_meta_url, dump_path)

            new_index_path = new_meta / backend.index_filename(new_engine)
            if new_engine == "redis":
                # Flush the loaded namespace to dump.rdb and stop the daemon
                # so we get a stable file to snapshot.
                await asyncio.to_thread(
                    backend.save_metadata,
                    new_engine,
                    redis_port=new_redis.port if new_redis else None,
                )
                backend.stop_metadata_engine(new_redis, timeout=10.0)
                new_redis = None
            else:
                await asyncio.to_thread(backend.checkpoint_metadata, str(new_index_path))

            if not new_index_path.exists():
                raise RuntimeError(f"juicefs load did not produce {new_index_path} — migration aborted.")

            # Disjoint the migrated engine's allocator space from the source
            # engine's. See class:`Volume.fork` for the rationale: even though
            # migration intends the source to be retired, defense in depth.
            offset = _random_fork_offset()
            _logger.info("[Volume.migrate_metadata_engine] disjoint counter offset=%d", offset)
            await asyncio.to_thread(backend.disjoint_fork_counters, str(new_index_path), new_engine, offset)

            snapshot_path = await asyncio.to_thread(backend.snapshot_index, new_index_path, new_engine, "vol-migrate-")
            cleanup_paths.append(snapshot_path)
            ctx = internal_ctx()
            index_bytes = os.path.getsize(snapshot_path)
            # No hash_method — see fork()'s upload site for why.
            new_file: File = await File.from_local(
                snapshot_path,
                remote_destination=ctx.raw_data.get_random_remote_path(file_name=backend.index_filename(new_engine)),
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
                backend.stop_metadata_engine(src_redis, timeout=5.0)
            if new_redis is not None:
                backend.stop_metadata_engine(new_redis, timeout=5.0)
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


def _random_fork_offset() -> int:
    """Return a random offset in ``[2³², 2³² + 2⁵⁶)`` — the *desired* fork
    offset, before any counter-headroom clamping.

    The lower bound (2³²) ensures every fork's allocator space is at least
    ~4B IDs ahead of the parent — JuiceFS allocates chunk-slice IDs in
    batches of 4096, so this is comfortably more than any realistic burst.
    The 56-bit upper bound keeps successive forks well-clear of uint64
    overflow under typical use; for deep fork chains the backend shrinks the
    actual applied offset based on the counter's remaining headroom.
    Birthday-collision probability for 10⁶ siblings is on the order of 10⁻⁵.
    """
    return (1 << 32) + secrets.randbits(56)


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


def volume_image_local(base: Optional[Image] = None) -> Image:
    """Image flavor for the ``local`` :class:`Volume` backend.

    The local backend only needs the ``tar`` binary, which ships with
    every Debian base. This helper exists for symmetry with
    :func:`volume_image` — call it whenever you would call ``volume_image``
    so callers can swap backends without restructuring image code.
    """
    if base is None:
        base = Image.from_debian_base(install_flyte=False)
    return base


def volume_pod_template_local(
    *,
    mount_path: str = _DEFAULT_MOUNT_PATH,
    primary_container_name: str = "primary",
) -> PodTemplate:
    """PodTemplate flavor for the ``local`` :class:`Volume` backend.

    Unlike :func:`volume_pod_template`, this template:

    * adds **only** an ``emptyDir`` for ``mount_path`` (no metadata dir, no
      chunk cache, no ``/dev/fuse``);
    * runs as the default container user, **not** privileged, with no
      added capabilities.

    Everything the local backend needs is a writable directory and the
    container's normal network access.
    """
    from kubernetes.client import (
        V1Container,
        V1EmptyDirVolumeSource,
        V1PodSpec,
        V1Volume,
        V1VolumeMount,
    )

    primary = V1Container(
        name=primary_container_name,
        volume_mounts=[V1VolumeMount(name="vol-workspace", mount_path=mount_path)],
    )
    pod_spec = V1PodSpec(
        containers=[primary],
        volumes=[V1Volume(name="vol-workspace", empty_dir=V1EmptyDirVolumeSource())],
    )
    return PodTemplate(pod_spec=pod_spec, primary_container_name=primary_container_name)


__all__ = [
    "Volume",
    "volume_image",
    "volume_image_local",
    "volume_pod_template",
    "volume_pod_template_local",
]
