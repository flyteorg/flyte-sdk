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
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict

from flyte._logging import logger as _logger
from flyte._pod import PodTemplate
from flyte.io._file import File

_DEFAULT_META_DIR = "/var/lib/flyte-volume"
_DEFAULT_CACHE_DIR = "/var/cache/flyte-volume"
_DEFAULT_MOUNT_PATH = "/workspace"
_INDEX_FILENAME = "index.db"
_CLIENT_BINARY = "juicefs"  # implementation detail
_CLIENT_VERSION = "1.1.2"


class Volume(BaseModel):
    """A persistent volume identified by its SQLite metadata index.

    A ``Volume`` is content-addressable lineage on top of an object-store
    bucket. The bucket holds the data chunks; the index (a SQLite DB) holds
    the entire namespace. Cloning the index produces an independent fork that
    initially sees the same file tree and shares chunk objects but diverges
    as either side writes.
    """

    name: str
    bucket: str
    storage: str = "s3"
    index: Optional[File] = None
    parent: Optional[File] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Track active mounts so commit() can wait on the daemon.
    _live_procs: ClassVar[Dict[str, subprocess.Popen]] = {}

    @classmethod
    def empty(cls, name: str, bucket: str, *, storage: str = "s3") -> "Volume":
        """Declare a brand-new volume. The first ``mount()`` call will
        bootstrap the namespace (the underlying client refuses to format
        over a non-empty bucket prefix).
        """
        return cls(name=name, bucket=bucket, storage=storage, index=None, parent=None)

    async def mount(
        self,
        *,
        mount_path: str = _DEFAULT_MOUNT_PATH,
        meta_dir: str = _DEFAULT_META_DIR,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        timeout: float = 120.0,
    ) -> None:
        """Format (if fresh) and mount the volume at ``mount_path`` in this
        process.

        Call once near the top of a task body before reading or writing under
        ``mount_path``.
        """
        meta = Path(meta_dir)
        meta.mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(mount_path).mkdir(parents=True, exist_ok=True)

        db_path = meta / _INDEX_FILENAME
        if self.index is not None:
            _logger.info("[Volume.mount] downloading index %s", self.index.path)
            await self.index.download(str(db_path))
        elif db_path.exists():
            # Stale index from an earlier mount in the same pod (shouldn't
            # happen in practice; tasks get fresh pods).
            db_path.unlink()

        meta_url = f"sqlite3://{db_path}"

        if self.index is None:
            _logger.info(
                "[Volume.mount] formatting fresh volume name=%s bucket=%s storage=%s",
                self.name, self.bucket, self.storage,
            )
            await asyncio.to_thread(
                _run_check,
                [_CLIENT_BINARY, "format", "--storage", self.storage, "--bucket", self.bucket,
                 meta_url, self.name],
            )

        _logger.info("[Volume.mount] mounting at %s", mount_path)
        proc = subprocess.Popen(  # noqa: ASYNC220 - long-lived daemon held in a class-level dict
            [_CLIENT_BINARY, "mount", "--cache-dir", cache_dir, meta_url, mount_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        type(self)._live_procs[mount_path] = proc

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if proc.poll() is not None:
                out = (proc.stdout.read().decode(errors="replace") if proc.stdout else "")
                raise RuntimeError(f"volume client exited prematurely (rc={proc.returncode}): {out}")
            if _is_fuse_mount(mount_path):
                break
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"mount at {mount_path} did not become a FUSE mountpoint within {timeout}s")
            await asyncio.sleep(0.5)

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
        # Flush page cache so any unfsync'd writes hit the filesystem first.
        await asyncio.to_thread(os.sync)

        meta = Path(meta_dir)
        db_path = meta / _INDEX_FILENAME

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

        # Force WAL pages back into the main .db before uploading.
        await asyncio.to_thread(_wal_checkpoint, str(db_path))

        new_index = await File.from_local(str(db_path))
        return Volume(
            name=self.name,
            bucket=self.bucket,
            storage=self.storage,
            index=new_index,
            parent=self.index,
        )

    async def fork(
        self,
        name: str,
        *,
        meta_dir: str = _DEFAULT_META_DIR,
    ) -> "Volume":
        """Snapshot the current live index via SQLite's online backup API and
        return a new ``Volume`` that points at the snapshot.

        Both the original and the fork reference the same bucket. Chunk keys
        are derived from inode IDs, which diverge from the fork point, so
        concurrent writes do not collide. Reads after the fork are namespace-
        isolated: writes from one side are invisible to the other.
        """
        src = Path(meta_dir) / _INDEX_FILENAME
        if not src.exists():
            raise RuntimeError(f"Cannot fork: no live index at {src}. Call mount() first.")

        await asyncio.to_thread(os.sync)
        await asyncio.to_thread(_wal_checkpoint, str(src))

        def _backup() -> str:
            tmp = tempfile.NamedTemporaryFile(prefix="vol-fork-", suffix=".db", delete=False)
            tmp.close()
            with sqlite3.connect(str(src)) as live, sqlite3.connect(tmp.name) as dst:
                live.backup(dst)
            return tmp.name

        snapshot_path = await asyncio.to_thread(_backup)
        try:
            new_index = await File.from_local(snapshot_path)
        finally:
            try:
                os.unlink(snapshot_path)
            except OSError:
                pass

        return Volume(
            name=name,
            bucket=self.bucket,
            storage=self.storage,
            index=new_index,
            parent=self.index,
        )


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


def volume_image(
    base: "object" = None,
    *,
    version: str = _CLIENT_VERSION,
    architecture: str = "amd64",
):
    """Layer the volume client and the ``fuse`` userspace tools onto ``base``
    (a :class:`flyte.Image`). Returns a new :class:`flyte.Image`.

    The primary container needs the volume client binary and the
    ``fusermount`` helper because :class:`Volume` mounts in-process rather
    than via a sidecar.
    """
    if base is None:
        import flyte

        base = flyte.Image.from_debian_base(install_flyte=False)

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
    return (
        base.with_apt_packages("ca-certificates", "curl", "fuse")
        .with_commands([install_cmd])
    )


def volume_pod_template(
    *,
    mount_path: str = _DEFAULT_MOUNT_PATH,
    meta_dir: str = _DEFAULT_META_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    primary_container_name: str = "primary",
) -> PodTemplate:
    """Build a :class:`flyte.PodTemplate` that lets the primary container
    mount a :class:`Volume` in-process.

    The returned template:

    * adds ``emptyDir`` volumes for the metadata directory, the chunk cache,
      and the mount point;
    * adds a ``hostPath`` volume for ``/dev/fuse``;
    * makes the primary container privileged with ``CAP_SYS_ADMIN`` so it
      can perform the FUSE mount itself.

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
            V1Volume(name="vol-cache", empty_dir=V1EmptyDirVolumeSource()),
            V1Volume(name="vol-workspace", empty_dir=V1EmptyDirVolumeSource()),
            V1Volume(
                name="fuse-device",
                host_path=V1HostPathVolumeSource(path="/dev/fuse", type="CharDevice"),
            ),
        ],
    )

    return PodTemplate(pod_spec=pod_spec, primary_container_name=primary_container_name)


__all__ = ["Volume", "volume_image", "volume_pod_template"]
