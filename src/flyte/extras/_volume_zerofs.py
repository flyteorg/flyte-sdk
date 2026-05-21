from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

from flyte._context import internal_ctx
from flyte._logging import logger as _logger
from flyte.io._file import File

_CLIENT_BINARY = "zerofs"
_INDEX_FILENAME = "zerofs-zfs-manifest.json"
_DEFAULT_NINEP_SOCKET = "zerofs.9p.sock"
_DEFAULT_NBD_SOCKET = "zerofs.nbd.sock"
_DEFAULT_RPC_SOCKET = "zerofs.rpc.sock"
_DEFAULT_CONTROL_MOUNT = "zerofs-control"
_DEFAULT_DEVICE_SIZE_GB = 64
_ZFS_FUSE_PROCS: list[subprocess.Popen] = []


@dataclass(frozen=True)
class _ZeroFSZFSManifest:
    storage_url: str
    pool: str
    dataset: str
    snapshot: str
    device_name: str
    device_size_gb: int
    checkpoint: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "backend": "zerofs",
                "metadata_engine": "zfs",
                "storage_url": self.storage_url,
                "pool": self.pool,
                "dataset": self.dataset,
                "snapshot": self.snapshot,
                "device_name": self.device_name,
                "device_size_gb": self.device_size_gb,
                "checkpoint": self.checkpoint,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> "_ZeroFSZFSManifest":
        data = json.loads(raw)
        if data.get("backend") != "zerofs" or data.get("metadata_engine") != "zfs":
            raise RuntimeError(f"{_INDEX_FILENAME} is not a ZeroFS/ZFS volume manifest")
        return cls(
            storage_url=str(data["storage_url"]),
            pool=str(data["pool"]),
            dataset=str(data["dataset"]),
            snapshot=str(data["snapshot"]),
            device_name=str(data["device_name"]),
            device_size_gb=int(data["device_size_gb"]),
            checkpoint=data.get("checkpoint"),
        )


@dataclass
class _ZeroFSZFSState:
    proc: subprocess.Popen
    config_path: str
    storage_url: str
    pool: str
    dataset: str
    device_name: str
    device_size_gb: int
    nbd_device: Optional[str]
    vdev_path: str
    control_mount: str
    mount_path: Optional[str]


class ZeroFSZFSVolumeBackend:
    """Experimental backend: ZeroFS object storage, NBD block device, ZFS filesystem.

    ZeroFS provides the S3-backed block device; ZFS provides copy-on-write
    snapshots and clones. A committed Volume publishes a small JSON manifest
    that names the ZeroFS storage URL, ZFS pool/dataset, and committed snapshot.
    """

    name = "zerofs"
    client_binary = _CLIENT_BINARY
    native_lifecycle = True

    _live: ClassVar[dict[str, _ZeroFSZFSState]] = {}

    def index_filename(self, engine: str) -> str:
        return _INDEX_FILENAME

    def meta_url(self, meta_dir: str, engine: str, *, redis_port: Optional[int] = None) -> str:
        return str(Path(meta_dir) / _INDEX_FILENAME)

    async def mount_volume(
        self,
        volume: Any,
        *,
        mount_path: str,
        meta_dir: str,
        cache_dir: str,
        timeout: float,
        writeback: bool,
        upload_delay: Optional[str],
        max_uploads: int,
        attr_cache: float,
        entry_cache: float,
        dir_entry_cache: float,
        read_only: bool,
    ) -> None:
        if volume._engine() != "zfs":
            raise ValueError("ZeroFS backend only supports metadata_engine='zfs'.")
        if read_only:
            raise NotImplementedError("ZeroFS/ZFS read-only mounts are not implemented yet.")

        Path(meta_dir).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(mount_path).mkdir(parents=True, exist_ok=True)

        manifest = await self._load_or_create_manifest(volume, meta_dir)
        work_dataset = self._work_dataset(manifest.pool, volume.name)
        state: Optional[_ZeroFSZFSState] = None
        try:
            state = await asyncio.to_thread(
                self._start_stack,
                manifest,
                meta_dir,
                cache_dir,
                mount_path,
                work_dataset,
                timeout,
            )
            self._live[meta_dir] = state
        except Exception:
            if state is not None:
                await asyncio.to_thread(self._stop_stack, state, timeout)
            raise

    async def commit_volume(self, volume: Any, *, mount_path: str, meta_dir: str, timeout: float) -> Any:
        state = self._state(meta_dir)
        await asyncio.to_thread(_sync_filesystem, mount_path)
        snapshot = self._snapshot_name("commit")
        await asyncio.to_thread(_run_check, ["zfs", "snapshot", f"{state.dataset}@{snapshot}"])
        checkpoint = await asyncio.to_thread(self._create_checkpoint, state, "commit")
        used_bytes = await asyncio.to_thread(self._dataset_used_bytes, state.dataset)
        new_index, index_bytes = await self._upload_manifest(
            volume,
            _ZeroFSZFSManifest(
                storage_url=state.storage_url,
                pool=state.pool,
                dataset=state.dataset,
                snapshot=snapshot,
                device_name=state.device_name,
                device_size_gb=state.device_size_gb,
                checkpoint=checkpoint,
            ),
        )
        await asyncio.to_thread(self._stop_stack, state, timeout)
        self._live.pop(meta_dir, None)
        return volume.__class__(
            name=volume.name,
            bucket=volume.bucket,
            storage=volume.storage,
            index=new_index,
            parent=volume.index,
            metadata_engine="zfs",
            volume_backend="zerofs",
            used_bytes=used_bytes,
            inode_count=None,
            index_bytes=index_bytes,
        )

    async def commit_inplace_volume(self, volume: Any, *, meta_dir: str) -> Any:
        state = self._state(meta_dir)
        await asyncio.to_thread(_sync_filesystem, state.mount_path or "/")
        snapshot = self._snapshot_name("checkpoint")
        await asyncio.to_thread(_run_check, ["zfs", "snapshot", f"{state.dataset}@{snapshot}"])
        checkpoint = await asyncio.to_thread(self._create_checkpoint, state, "checkpoint")
        used_bytes = await asyncio.to_thread(self._dataset_used_bytes, state.dataset)
        new_index, index_bytes = await self._upload_manifest(
            volume,
            _ZeroFSZFSManifest(
                storage_url=state.storage_url,
                pool=state.pool,
                dataset=state.dataset,
                snapshot=snapshot,
                device_name=state.device_name,
                device_size_gb=state.device_size_gb,
                checkpoint=checkpoint,
            ),
        )
        return volume.__class__(
            name=volume.name,
            bucket=volume.bucket,
            storage=volume.storage,
            index=new_index,
            parent=volume.index,
            metadata_engine="zfs",
            volume_backend="zerofs",
            used_bytes=used_bytes,
            inode_count=None,
            index_bytes=index_bytes,
        )

    async def fork_volume(self, volume: Any, *, name: str, mount_path: str, meta_dir: str, timeout: float) -> Any:
        if meta_dir in self._live:
            state = self._live[meta_dir]
            parent_snapshot = self._snapshot_name("fork-parent")
            child_dataset = self._work_dataset(state.pool, name)
            child_snapshot = self._snapshot_name("fork-child")
            await asyncio.to_thread(_run_check, ["zfs", "snapshot", f"{state.dataset}@{parent_snapshot}"])
            await asyncio.to_thread(
                _run_check,
                ["zfs", "clone", "-p", f"{state.dataset}@{parent_snapshot}", child_dataset],
            )
            await asyncio.to_thread(_run_check, ["zfs", "snapshot", f"{child_dataset}@{child_snapshot}"])
            checkpoint = await asyncio.to_thread(self._create_checkpoint, state, "fork")
            used_bytes = await asyncio.to_thread(self._dataset_used_bytes, child_dataset)
            new_index, index_bytes = await self._upload_manifest(
                volume,
                _ZeroFSZFSManifest(
                    storage_url=state.storage_url,
                    pool=state.pool,
                    dataset=child_dataset,
                    snapshot=child_snapshot,
                    device_name=state.device_name,
                    device_size_gb=state.device_size_gb,
                    checkpoint=checkpoint,
                ),
            )
        else:
            if volume.index is None:
                raise RuntimeError("Cannot fork: ZeroFS/ZFS Volume is not mounted and has no manifest to fork from.")
            manifest = await self._load_manifest(volume, meta_dir)
            child_dataset = self._work_dataset(manifest.pool, name)
            child_snapshot = self._snapshot_name("fork-child")
            state = await asyncio.to_thread(
                self._start_stack,
                manifest,
                meta_dir,
                str(Path(meta_dir) / "cache"),
                None,
                None,
                timeout,
            )
            try:
                await asyncio.to_thread(
                    _run_check,
                    ["zfs", "clone", "-p", f"{manifest.dataset}@{manifest.snapshot}", child_dataset],
                )
                await asyncio.to_thread(_run_check, ["zfs", "snapshot", f"{child_dataset}@{child_snapshot}"])
                checkpoint = await asyncio.to_thread(self._create_checkpoint, state, "fork")
                used_bytes = await asyncio.to_thread(self._dataset_used_bytes, child_dataset)
                new_index, index_bytes = await self._upload_manifest(
                    volume,
                    _ZeroFSZFSManifest(
                        storage_url=state.storage_url,
                        pool=state.pool,
                        dataset=child_dataset,
                        snapshot=child_snapshot,
                        device_name=state.device_name,
                        device_size_gb=state.device_size_gb,
                        checkpoint=checkpoint,
                    ),
                )
            finally:
                await asyncio.to_thread(self._stop_stack, state, timeout)

        return volume.__class__(
            name=name,
            bucket=volume.bucket,
            storage=volume.storage,
            index=new_index,
            parent=volume.index,
            metadata_engine="zfs",
            volume_backend="zerofs",
            used_bytes=used_bytes,
            inode_count=None,
            index_bytes=index_bytes,
        )

    def _start_stack(
        self,
        manifest: _ZeroFSZFSManifest,
        meta_dir: str,
        cache_dir: str,
        mount_path: Optional[str],
        work_dataset: Optional[str],
        timeout: float,
    ) -> _ZeroFSZFSState:
        meta = Path(meta_dir)
        cache = Path(cache_dir)
        control_mount = str(meta / _DEFAULT_CONTROL_MOUNT)
        ninep_socket = str(meta / _DEFAULT_NINEP_SOCKET)
        nbd_socket = str(meta / _DEFAULT_NBD_SOCKET)
        rpc_socket = str(meta / _DEFAULT_RPC_SOCKET)
        config_path = str(meta / "zerofs.toml")

        Path(control_mount).mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        self._write_config(
            config_path=config_path,
            storage_url=manifest.storage_url,
            cache_dir=str(cache),
            ninep_socket=ninep_socket,
            nbd_socket=nbd_socket,
            rpc_socket=rpc_socket,
        )
        nbd_available = _ensure_kernel_devices()

        proc = subprocess.Popen(
            [self.client_binary, "run", "-c", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = _ZeroFSZFSState(
            proc=proc,
            config_path=config_path,
            storage_url=manifest.storage_url,
            pool=manifest.pool,
            dataset=work_dataset or manifest.dataset,
            device_name=manifest.device_name,
            device_size_gb=manifest.device_size_gb,
            nbd_device=None,
            vdev_path="",
            control_mount=control_mount,
            mount_path=mount_path,
        )
        try:
            _wait_for_socket(proc, ninep_socket, timeout)
            _run_check(
                [
                    "mount",
                    "-t",
                    "9p",
                    "-o",
                    "trans=unix,version=9p2000.L,cache=mmap,access=user",
                    ninep_socket,
                    control_mount,
                ]
            )
            if nbd_available:
                _wait_for_socket(proc, nbd_socket, timeout)
                state.nbd_device = _find_free_nbd_device()
                nbd_file = Path(control_mount) / ".nbd" / manifest.device_name
                nbd_file.parent.mkdir(parents=True, exist_ok=True)
                if not nbd_file.exists():
                    _run_check(["truncate", "-s", f"{manifest.device_size_gb}G", str(nbd_file)])
                _run_check(
                    [
                        "nbd-client",
                        "-unix",
                        nbd_socket,
                        state.nbd_device,
                        "-N",
                        manifest.device_name,
                        "-persist",
                        "-timeout",
                        "600",
                        "-connections",
                        "4",
                    ]
                )
                state.vdev_path = state.nbd_device
            else:
                vdev_file = Path(control_mount) / ".zfs-vdevs" / f"{manifest.device_name}.img"
                vdev_file.parent.mkdir(parents=True, exist_ok=True)
                if not vdev_file.exists():
                    _run_check(["truncate", "-s", f"{manifest.device_size_gb}G", str(vdev_file)])
                state.vdev_path = str(vdev_file)

            import_search_dir = None if state.nbd_device else str(Path(state.vdev_path).parent)
            if not self._pool_import(manifest.pool, search_dir=import_search_dir):
                if manifest.snapshot:
                    raise RuntimeError(f"Could not import existing ZFS pool {manifest.pool!r} for ZeroFS volume.")
                _run_check(["zpool", "create", "-f", "-m", "none", manifest.pool, state.vdev_path])

            if work_dataset is not None:
                if manifest.snapshot:
                    _run_check(["zfs", "clone", "-p", f"{manifest.dataset}@{manifest.snapshot}", work_dataset])
                else:
                    _run_check(["zfs", "create", "-p", work_dataset])

            if mount_path is not None:
                _run_check(["zfs", "set", f"mountpoint={mount_path}", state.dataset])
                _run_check(["zfs", "mount", state.dataset])
            return state
        except Exception:
            self._stop_stack(state, timeout)
            raise

    def _stop_stack(self, state: _ZeroFSZFSState, timeout: float) -> None:
        if state.mount_path is not None:
            subprocess.run(["zfs", "unmount", state.dataset], capture_output=True, check=False)
        subprocess.run(["zpool", "export", state.pool], capture_output=True, check=False)
        if state.nbd_device:
            subprocess.run(["nbd-client", "-d", state.nbd_device], capture_output=True, check=False)
        subprocess.run(["umount", state.control_mount], capture_output=True, check=False)
        if state.proc.poll() is None:
            state.proc.terminate()
            try:
                state.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state.proc.kill()
                state.proc.wait(5)

    def _pool_import(self, pool: str, *, search_dir: Optional[str] = None) -> bool:
        cmd = ["zpool", "import", "-N", "-f"]
        if search_dir is not None:
            cmd.extend(["-d", search_dir])
        cmd.append(pool)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True
        _logger.info("[Volume.zerofs] zpool import failed, will create if this is a fresh volume: %s", result.stderr)
        return False

    def _create_checkpoint(self, state: _ZeroFSZFSState, prefix: str) -> str:
        checkpoint = self._snapshot_name(prefix)
        _run_check([self.client_binary, "checkpoint", "create", "-c", state.config_path, checkpoint])
        return checkpoint

    def _dataset_used_bytes(self, dataset: str) -> Optional[int]:
        result = subprocess.run(
            ["zfs", "get", "-Hp", "-o", "value", "used", dataset],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    async def _load_or_create_manifest(self, volume: Any, meta_dir: str) -> _ZeroFSZFSManifest:
        if volume.index is not None:
            return await self._load_manifest(volume, meta_dir)
        storage_url = self._storage_url(volume)
        pool = self._pool_name(volume.name, storage_url)
        return _ZeroFSZFSManifest(
            storage_url=storage_url,
            pool=pool,
            dataset=f"{pool}/root",
            snapshot="",
            device_name=self._device_name(volume.name),
            device_size_gb=int(os.environ.get("FLYTE_ZEROVOL_SIZE_GB", _DEFAULT_DEVICE_SIZE_GB)),
        )

    async def _load_manifest(self, volume: Any, meta_dir: str) -> _ZeroFSZFSManifest:
        if volume.index is None:
            raise RuntimeError("ZeroFS/ZFS Volume has no manifest.")
        manifest_path = Path(meta_dir) / _INDEX_FILENAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        await volume.index.download(str(manifest_path))
        return _ZeroFSZFSManifest.from_json(manifest_path.read_text())

    async def _upload_manifest(self, volume: Any, manifest: _ZeroFSZFSManifest) -> tuple[File, int]:
        with tempfile.NamedTemporaryFile(prefix="vol-zerofs-", suffix=".json", mode="w", delete=False) as f:
            f.write(manifest.to_json())
            path = f.name
        try:
            index_bytes = os.path.getsize(path)
            ctx = internal_ctx()
            new_file: File = await File.from_local(
                path,
                remote_destination=ctx.raw_data.get_random_remote_path(file_name=_INDEX_FILENAME),
            )
            return new_file, index_bytes
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _write_config(
        self,
        *,
        config_path: str,
        storage_url: str,
        cache_dir: str,
        ninep_socket: str,
        nbd_socket: str,
        rpc_socket: str,
        cache_disk_size_gb: int = 16,
    ) -> None:
        if os.environ.get("FLYTE_ZEROVOL_PASSWORD"):
            password_var = "FLYTE_ZEROVOL_PASSWORD"
        elif os.environ.get("ZEROFS_PASSWORD"):
            password_var = "ZEROFS_PASSWORD"
        else:
            raise RuntimeError("ZeroFS/ZFS volumes require FLYTE_ZEROVOL_PASSWORD or ZEROFS_PASSWORD to be set.")

        lines = [
            "[cache]",
            f'dir = "{cache_dir}"',
            f"disk_size_gb = {cache_disk_size_gb}",
            "",
            "[storage]",
            f'url = "{storage_url}"',
            f'encryption_password = "${{{password_var}}}"',
            "",
            "[servers.ninep]",
            f'unix_socket = "{ninep_socket}"',
            "",
            "[servers.nbd]",
            f'unix_socket = "{nbd_socket}"',
            "",
            "[servers.rpc]",
            f'unix_socket = "{rpc_socket}"',
            "",
        ]
        if storage_url.startswith("s3://"):
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            lines += ["[aws]"]
            if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
                lines += [
                    'access_key_id = "${AWS_ACCESS_KEY_ID}"',
                    'secret_access_key = "${AWS_SECRET_ACCESS_KEY}"',
                ]
            if region:
                lines.append(f'default_region = "{region}"')
            lines.append("")
        Path(config_path).write_text("\n".join(lines))

    def _storage_url(self, volume: Any) -> str:
        return f"{str(volume.bucket).rstrip('/')}/{volume.name}"

    def _snapshot_name(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def _work_dataset(self, pool: str, name: str) -> str:
        return f"{pool}/{_safe_zfs_component(name)}-{uuid.uuid4().hex[:12]}"

    def _pool_name(self, name: str, storage_url: str) -> str:
        digest = hashlib.sha1(f"{name}:{storage_url}".encode()).hexdigest()[:16]
        return f"flytevol{digest}"

    def _device_name(self, name: str) -> str:
        return _safe_zfs_component(name)

    def _state(self, meta_dir: str) -> _ZeroFSZFSState:
        try:
            return self._live[meta_dir]
        except KeyError as e:
            raise RuntimeError("ZeroFS/ZFS Volume is not mounted in this process.") from e


def _safe_zfs_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-")
    return cleaned or "volume"


def _run_check(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _sync_filesystem(path: str) -> None:
    syncfs = getattr(os, "syncfs", None)
    if syncfs is not None:
        fd: Optional[int] = None
        try:
            fd = os.open(path, os.O_RDONLY)
            syncfs(fd)
            return
        except OSError:
            pass
        finally:
            if fd is not None:
                os.close(fd)
    os.sync()


def _wait_for_socket(proc: subprocess.Popen, path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"zerofs exited prematurely (rc={proc.returncode}): {out}")
        if Path(path).exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"zerofs did not create {path} within {timeout}s")
        time.sleep(0.2)


def _find_free_nbd_device() -> str:
    requested = os.environ.get("FLYTE_ZEROVOL_NBD_DEVICE")
    if requested:
        return requested
    for idx in range(64):
        dev = Path(f"/dev/nbd{idx}")
        if not dev.exists():
            continue
        pid = Path(f"/sys/block/nbd{idx}/pid")
        try:
            if not pid.exists() or pid.read_text().strip() in ("", "0"):
                return str(dev)
        except OSError:
            continue
    raise RuntimeError("No free /dev/nbd* device is available for ZeroFS/ZFS volume mount.")


def _ensure_kernel_devices() -> bool:
    _modprobe("nbd", "max_part=0", "nbds_max=64")
    _ensure_9p_support()
    nbd_available = _ensure_nbd_device_nodes()
    _ensure_zfs_available()
    return nbd_available


def _ensure_9p_support() -> None:
    """zerofs talks to its control socket over 9p. Ensure the filesystem
    type is registered in the kernel, surfacing a useful error if not."""
    for mod in ("9pnet", "9pnet_virtio", "9p"):
        _modprobe(mod)
    try:
        registered = "9p" in Path("/proc/filesystems").read_text()
    except OSError as e:
        raise RuntimeError(f"cannot read /proc/filesystems: {e}") from e
    if not registered:
        try:
            uname = subprocess.run(["uname", "-r"], capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            uname = "?"
        try:
            mod_dir = Path(f"/lib/modules/{uname}/kernel/fs/9p")
            mods = sorted(p.name for p in mod_dir.iterdir()) if mod_dir.exists() else []
        except OSError:
            mods = []
        raise RuntimeError(
            f"9p filesystem support is not available in the kernel (uname={uname!r}). "
            f"Modules found in {mod_dir if mods else '/lib/modules/.../fs/9p'}: {mods!r}. "
            "ZeroFS requires the 9p kernel module; either the host kernel was built without it "
            "or /lib/modules isn't mounted from a kernel that provides it."
        )


def _ensure_zfs_available() -> None:
    if shutil.which("zfs-fuse"):
        _start_zfs_fuse()
        return
    _modprobe("zfs")
    _ensure_zfs_device_node()


def _start_zfs_fuse() -> None:
    pidfile = "/tmp/flyte-zfs-fuse.pid"
    if _pidfile_is_running(pidfile):
        return

    Path("/var/lock/zfs").mkdir(parents=True, exist_ok=True)

    # Probe the binary first — surfaces "command not found" / missing libs.
    # zfs-fuse --help exits with rc=64 (EX_USAGE) but still prints its usage
    # banner, so the only thing we check here is that *some* output mentions
    # zfs.
    probe = subprocess.run(
        ["zfs-fuse", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    probe_out = (probe.stdout or "") + (probe.stderr or "")
    if "zfs" not in probe_out.lower():
        raise RuntimeError(
            f"zfs-fuse --help did not produce a recognizable banner "
            f"(rc={probe.returncode}): stdout={probe.stdout!r} stderr={probe.stderr!r}"
        )

    # Container env state in case the daemon dies silently — zfs-fuse may
    # log to syslog (unavailable in containers), so capture surrounding
    # state up-front for the error path.
    def _diag() -> str:
        bits: list[str] = []
        for path in ("/dev/fuse", "/var/lock/zfs", pidfile):
            try:
                st = os.stat(path)
                bits.append(f"{path}=mode=0o{st.st_mode:o}")
            except OSError as e:
                bits.append(f"{path}={e}")
        try:
            with open("/proc/filesystems") as f:
                bits.append("fuse_in_/proc/filesystems=" + str("fuse" in f.read()))
        except OSError as e:
            bits.append(f"/proc/filesystems={e}")
        return "; ".join(bits)

    log_path = "/tmp/flyte-zfs-fuse.log"
    # Spawn via /bin/sh so we can capture pre-exec failures and the final
    # exit code. zfs-fuse in DAEMONIZED mode (no --no-daemon) writes the
    # pidfile after init completes; in --no-daemon mode some builds skip
    # the pidfile entirely, which leaves us with no readiness signal.
    # Stay daemonized — the daemon detaches and the sh wrapper exits
    # quickly with the daemon's startup exit code.
    cmd = (
        f"set +e; "
        f"echo '== diag ==' >>{log_path}; "
        f"id >>{log_path} 2>&1; "
        f"ls -la /dev/fuse /var/lock/zfs >>{log_path} 2>&1; "
        f"ldd $(command -v zfs-fuse) >>{log_path} 2>&1 || true; "
        f"echo '== launching ==' >>{log_path}; "
        f"zfs-fuse --pidfile {pidfile} >>{log_path} 2>&1; "
        f'echo "__exit=$?" >>{log_path}'
    )
    proc = subprocess.Popen(
        ["/bin/sh", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def _log_tail() -> str:
        try:
            return Path(log_path).read_text(errors="replace")[-4096:]
        except OSError:
            return ""

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _pidfile_is_running(pidfile):
            _ZFS_FUSE_PROCS.append(proc)
            return
        if proc.poll() is not None:
            # sh exited. In daemonized mode this is normal — the daemon
            # detached. Check the pidfile a few more times before failing.
            for _ in range(20):
                if _pidfile_is_running(pidfile):
                    _ZFS_FUSE_PROCS.append(proc)
                    return
                time.sleep(0.2)
            raise RuntimeError(f"zfs-fuse failed to start (sh rc={proc.returncode}) log={_log_tail()!r} diag={_diag()}")
        time.sleep(0.2)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5)
    raise TimeoutError(f"zfs-fuse did not start within 30s; log={_log_tail()!r} diag={_diag()}")


def _pidfile_is_running(pidfile: str) -> bool:
    try:
        pid = int(Path(pidfile).read_text().strip())
    except (OSError, ValueError):
        return False
    return Path(f"/proc/{pid}").exists()


def _modprobe(module: str, *args: str) -> None:
    result = subprocess.run(["modprobe", module, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        _logger.info("[Volume.zerofs] modprobe %s failed; checking for existing support: %s", module, result.stderr)


def _ensure_nbd_device_nodes() -> bool:
    major = _device_major("Block devices:", "nbd")
    if major is None:
        _logger.info("[Volume.zerofs] kernel NBD support is unavailable; using a ZeroFS-backed ZFS file vdev")
        return False
    for idx in range(64):
        _ensure_device_node(Path(f"/dev/nbd{idx}"), stat.S_IFBLK, major, idx)
    return True


def _ensure_zfs_device_node() -> None:
    minor = _misc_minor("zfs")
    if minor is None:
        raise RuntimeError("Kernel ZFS support is unavailable; modprobe zfs did not expose /proc/misc entry 'zfs'.")
    _ensure_device_node(Path("/dev/zfs"), stat.S_IFCHR, 10, minor)


def _ensure_device_node(path: Path, kind: int, major: int, minor: int) -> None:
    try:
        existing = path.stat()
    except FileNotFoundError:
        os.mknod(path, kind | 0o660, os.makedev(major, minor))
        return
    if (
        stat.S_IFMT(existing.st_mode) != kind
        or os.major(existing.st_rdev) != major
        or os.minor(existing.st_rdev) != minor
    ):
        raise RuntimeError(f"{path} exists but is not the expected device node ({major}:{minor}).")


def _device_major(section: str, name: str) -> Optional[int]:
    current_section = None
    try:
        lines = Path("/proc/devices").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.endswith("devices:"):
            current_section = line
            continue
        if current_section != section:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return int(parts[0])
    return None


def _misc_minor(name: str) -> Optional[int]:
    try:
        lines = Path("/proc/misc").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return int(parts[0])
    return None
