from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from flyte._logging import logger as _logger
from flyte.extras._volume_backend import _MetadataEngineState, _MountConfig

_SQLITE_INDEX_FILENAME = "index.db"
_REDIS_INDEX_FILENAME = "dump.rdb"
_CLIENT_BINARY = "juicefs"
# Pinned to a release whose SQLite ``jfs_counter`` and Redis ``next{chunk,inode,session}``
# keys are known to match what :func:`_disjoint_fork_counters` expects.
# Verified against 1.1.2 and 1.3.1 — bumping further requires re-validating
# the counter schema (see tests/flyte/extras/test_volume.py::TestForkIsolationRegression).
_CLIENT_VERSION = "1.3.1"
_REDIS_PORT = 6379


class JuiceFSVolumeBackend:
    name = "juicefs"
    client_binary = _CLIENT_BINARY

    def index_filename(self, engine: str) -> str:
        return _REDIS_INDEX_FILENAME if engine == "redis" else _SQLITE_INDEX_FILENAME

    def meta_url(self, meta_dir: str, engine: str, *, redis_port: Optional[int] = None) -> str:
        if engine == "redis":
            return f"redis://127.0.0.1:{redis_port or _REDIS_PORT}/0"
        return f"sqlite3://{Path(meta_dir) / _SQLITE_INDEX_FILENAME}"

    async def start_metadata_engine(self, meta_dir: str, engine: str) -> Optional[_MetadataEngineState]:
        if engine == "redis":
            return await _start_redis(meta_dir)
        return None

    def stop_metadata_engine(self, state: Optional[_MetadataEngineState], timeout: float) -> None:
        _stop_redis(state, timeout)

    def format(self, *, storage: str, bucket: str, meta_url: str, name: str) -> None:
        _run_check([self.client_binary, "format", "--storage", storage, "--bucket", bucket, meta_url, name])

    def dump_metadata(self, meta_url: str, dump_path: str) -> None:
        _run_check([self.client_binary, "dump", meta_url, dump_path])

    def load_metadata(self, meta_url: str, dump_path: str) -> None:
        _run_check([self.client_binary, "load", meta_url, dump_path])

    def mount_cmd(self, *, config: _MountConfig, engine: str, mount_path: str) -> list[str]:
        cmd = [
            self.client_binary,
            "mount",
            "--cache-dir",
            config.cache_dir,
            "--max-uploads",
            str(config.max_uploads),
            "--attr-cache",
            str(config.attr_cache),
            "--entry-cache",
            str(config.entry_cache),
            "--dir-entry-cache",
            str(config.dir_entry_cache),
        ]
        if config.read_only:
            # JuiceFS read-only mount: writes through the FUSE mount return
            # EROFS. Implies writeback is moot (no chunks to flush).
            cmd.append("--read-only")
        elif config.writeback:
            cmd.append("--writeback")
            if config.upload_delay:
                cmd += ["--upload-delay", config.upload_delay]
        cmd += [self.meta_url(config.meta_dir, engine, redis_port=config.redis_port), mount_path]
        return cmd

    def is_mounted(self, mount_path: str) -> bool:
        return _is_fuse_mount(mount_path)

    def unmount(self, mount_path: str, *, flush: bool = False) -> None:
        cmd = [self.client_binary, "umount"]
        if flush:
            cmd.append("--flush")
        cmd.append(mount_path)
        _run_check(cmd)

    def sync_filesystem(self, path: str) -> None:
        _sync_filesystem(path)

    def save_metadata(self, engine: str, *, redis_port: Optional[int] = None) -> None:
        if engine == "redis":
            _redis_save(redis_port or _REDIS_PORT)

    def checkpoint_metadata(self, index_path: str) -> None:
        _wal_checkpoint(index_path)

    def snapshot_index(self, src: Path, engine: str, tmp_prefix: str) -> str:
        if engine == "redis":
            import shutil

            tmp = tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=".rdb", delete=False)
            tmp.close()
            shutil.copyfile(str(src), tmp.name)
            return tmp.name

        tmp = tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=".db", delete=False)
        tmp.close()
        with sqlite3.connect(str(src)) as live, sqlite3.connect(tmp.name) as dst:
            live.backup(dst)
        return tmp.name

    async def query_stats(
        self, meta_dir: str, engine: str, *, redis_port: Optional[int] = None
    ) -> tuple[Optional[int], Optional[int]]:
        return await _query_volume_stats(meta_dir, engine, redis_port=redis_port)

    def disjoint_fork_counters(self, index_path: str, engine: str, offset: int) -> int:
        return _disjoint_fork_counters(index_path, engine, offset)


def _index_filename(engine: str) -> str:
    return JuiceFSVolumeBackend().index_filename(engine)


def _meta_url(meta_dir: str, engine: str, *, redis_port: Optional[int] = None) -> str:
    return JuiceFSVolumeBackend().meta_url(meta_dir, engine, redis_port=redis_port)


async def _query_volume_stats(
    meta_dir: str, engine: str, *, redis_port: Optional[int] = None
) -> tuple[Optional[int], Optional[int]]:
    """Best-effort: return ``(used_bytes, inode_count)`` for the volume backed
    by ``meta_dir`` / ``engine``. Shells out to ``juicefs status`` and parses
    the JSON. Returns ``(None, None)`` on any failure so commit/fork are never
    blocked by stats collection.
    """
    meta_url = _meta_url(meta_dir, engine, redis_port=redis_port)
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
    """True if *path* is a FUSE mount (not just any mountpoint)."""
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


def _sync_filesystem(path: str) -> None:
    """Flush the filesystem containing path, falling back to process-wide sync."""
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


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _start_redis(meta_dir: str, timeout: float = 30.0, port: Optional[int] = None) -> _MetadataEngineState:
    """Spawn an in-process ``redis-server`` rooted at ``meta_dir``."""
    port = port or _free_tcp_port()
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
            ["redis-cli", "-p", str(port), "ping"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and "PONG" in r.stdout:
            _logger.info("[Volume.mount] redis-server ready on port %d", port)
            return _MetadataEngineState(proc=proc, port=port)
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"redis-server did not become ready within {timeout}s")
        await asyncio.sleep(0.2)


def _redis_save(port: int = _REDIS_PORT) -> None:
    """Trigger a synchronous RDB save."""
    _run_check(["redis-cli", "-p", str(port), "SAVE"])


def _stop_redis(state: Optional[_MetadataEngineState], timeout: float) -> None:
    """Shut down a redis-server cleanly. Caller must have already saved if
    they want the in-memory state persisted.
    """
    if state is None:
        return
    # SHUTDOWN NOSAVE — we drove persistence ourselves via SAVE.
    subprocess.run(
        ["redis-cli", "-p", str(state.port), "SHUTDOWN", "NOSAVE"],
        capture_output=True,
        check=False,
    )
    try:
        state.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _logger.warning("[Volume.commit] redis-server didn't exit in %ss; killing", timeout)
        state.proc.kill()
        state.proc.wait(5)


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


_FORK_COUNTER_NAMES_SQLITE: tuple = ("nextChunk", "nextInode", "nextSession")
_FORK_COUNTER_NAMES_REDIS: tuple = ("nextchunk", "nextinode", "nextsession")
_COUNTER_MAX = (1 << 63) - 1
_MIN_FORK_OFFSET = 1 << 32


def _safe_fork_offset(current_max: int, desired: int) -> int:
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
    if offset <= 0:
        raise ValueError(f"offset must be positive, got {offset}")
    rdb = Path(rdb_path)
    if not rdb.exists():
        raise RuntimeError(f"{rdb_path} does not exist")

    port = _free_tcp_port()
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
    if engine == "redis":
        return _disjoint_counters_redis(index_path, offset)
    return _disjoint_counters_sqlite(index_path, offset)
