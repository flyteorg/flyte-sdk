"""Tests for flyte.extras Volume.

Layered:
1. Pure-Python helpers (always run): counter-bump, _meta_url, _index_filename,
   _is_fuse_mount, _random_fork_offset.
2. JuiceFS integration (skipped without juicefs binary / FUSE / redis-server):
   real format+mount+commit+fork round-trips, including the fork-isolation
   regression that motivated the counter-bump fix.
"""

from __future__ import annotations

import itertools
import shutil
import socket
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest

import flyte.extras._volume as volume_mod
import flyte.extras._volume_juicefs as juicefs_mod
from flyte.extras._volume import (
    Volume,
    _client_bucket_uri,
    _default_bucket,
    _random_fork_offset,
)
from flyte.extras._volume_backend import _MetadataEngineState, _MountConfig
from flyte.extras._volume_juicefs import (
    _COUNTER_MAX,
    _MIN_FORK_OFFSET,
    _disjoint_counters_redis,
    _disjoint_counters_sqlite,
    _disjoint_fork_counters,
    _index_filename,
    _is_fuse_mount,
    _meta_url,
    _safe_fork_offset,
    _wal_checkpoint,
)
from flyte.io._file import File


def _snapshot_sqlite_db(src_db: Path, dst_db: Path) -> None:
    """Mirror what Volume.fork() does: WAL-checkpoint the source so the main
    .db file contains all committed transactions, then use SQLite's online
    backup API to produce a consistent copy.

    ``shutil.copyfile`` is not safe here because JuiceFS leaves the WAL
    sidecar (``db-wal``) on disk after umount; copying the main file alone
    yields a half-formatted view of the volume.
    """
    _wal_checkpoint(str(src_db))
    with sqlite3.connect(str(src_db)) as src, sqlite3.connect(str(dst_db)) as dst:
        src.backup(dst)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


HAS_JUICEFS = _which("juicefs") is not None
HAS_REDIS = _which("redis-server") is not None and _which("redis-cli") is not None
HAS_FUSE = Path("/dev/fuse").exists()

skip_no_juicefs = pytest.mark.skipif(not HAS_JUICEFS, reason="juicefs binary not in PATH")
skip_no_redis = pytest.mark.skipif(not HAS_REDIS, reason="redis-server/redis-cli not in PATH")
skip_no_fuse = pytest.mark.skipif(not HAS_FUSE, reason="/dev/fuse not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SQLITE_COUNTER_DEFAULTS = {
    "nextInode": 2050,
    "nextChunk": 8193,
    "nextSession": 3,
    "usedSpace": 8192,
    "totalInodes": 2,
    "nextCleanupSlices": 0,
}


def _make_sqlite_index(tmp_path: Path, counters: Optional[dict] = None) -> Path:
    """Build a minimal JuiceFS-shaped SQLite metadata file for unit tests."""
    p = tmp_path / "index.db"
    counters = counters if counters is not None else _SQLITE_COUNTER_DEFAULTS
    with sqlite3.connect(str(p)) as conn:
        conn.execute("CREATE TABLE jfs_counter (name TEXT PRIMARY KEY, value INTEGER)")
        conn.executemany("INSERT INTO jfs_counter VALUES(?, ?)", counters.items())
        conn.commit()
    return p


def _read_sqlite_counters(p: Path) -> dict:
    with sqlite3.connect(str(p)) as conn:
        return dict(conn.execute("SELECT name, value FROM jfs_counter").fetchall())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


# ---------------------------------------------------------------------------
# 1. Unit tests — counter bump helpers (pure Python; no JuiceFS needed)
# ---------------------------------------------------------------------------


class TestRandomForkOffset:
    def test_in_range(self):
        for _ in range(100):
            o = _random_fork_offset()
            assert (1 << 32) <= o < ((1 << 32) + (1 << 56))

    def test_distinct_across_calls(self):
        # 56 bits of entropy — 100 draws should never collide.
        seen = {_random_fork_offset() for _ in range(100)}
        assert len(seen) == 100


class TestSafeForkOffset:
    """The overflow guard. Verifies clamping behavior at boundaries."""

    def test_no_clamp_when_plenty_of_headroom(self):
        # Counter very small; offset returned unchanged.
        desired = 1 << 40
        assert _safe_fork_offset(current_max=10_000, desired=desired) == desired

    def test_clamps_to_half_headroom_near_max(self):
        # Headroom = _COUNTER_MAX - current_max
        current_max = _COUNTER_MAX - (1 << 40)
        applied = _safe_fork_offset(current_max=current_max, desired=1 << 56)
        # Should be no more than headroom // 2.
        assert applied <= (_COUNTER_MAX - current_max) // 2
        # And clamp must reduce desired since headroom was smaller than it.
        assert applied < (1 << 56)

    def test_returns_desired_when_smaller_than_half_headroom(self):
        # If desired fits well within half of headroom, return as-is.
        current_max = _COUNTER_MAX - (1 << 50)
        desired = 1 << 33
        assert _safe_fork_offset(current_max, desired) == desired

    def test_raises_when_exhausted(self):
        # Headroom < 2 * MIN_FORK_OFFSET → unsafe.
        current_max = _COUNTER_MAX - _MIN_FORK_OFFSET
        with pytest.raises(RuntimeError, match="exhausted its ID space"):
            _safe_fork_offset(current_max, _MIN_FORK_OFFSET)

    def test_rejects_non_positive_desired(self):
        with pytest.raises(ValueError, match="must be positive"):
            _safe_fork_offset(current_max=0, desired=0)
        with pytest.raises(ValueError, match="must be positive"):
            _safe_fork_offset(current_max=0, desired=-1)


class TestDisjointCountersSqlite:
    def test_bumps_only_allocator_counters(self, tmp_path):
        p = _make_sqlite_index(tmp_path)
        _disjoint_counters_sqlite(str(p), 1_000_000)
        after = _read_sqlite_counters(p)
        assert after == {
            "nextInode": 1_002_050,
            "nextChunk": 1_008_193,
            "nextSession": 1_000_003,
            "usedSpace": 8192,  # untouched
            "totalInodes": 2,  # untouched
            "nextCleanupSlices": 0,  # untouched
        }

    def test_idempotent_double_bump_is_additive(self, tmp_path):
        p = _make_sqlite_index(tmp_path)
        _disjoint_counters_sqlite(str(p), 1000)
        _disjoint_counters_sqlite(str(p), 2000)
        after = _read_sqlite_counters(p)
        assert after["nextChunk"] == 8193 + 3000
        assert after["nextInode"] == 2050 + 3000
        assert after["nextSession"] == 3 + 3000

    def test_rejects_non_positive_offset(self, tmp_path):
        p = _make_sqlite_index(tmp_path)
        with pytest.raises(ValueError):
            _disjoint_counters_sqlite(str(p), 0)
        with pytest.raises(ValueError):
            _disjoint_counters_sqlite(str(p), -1)

    def test_rejects_non_juicefs_db(self, tmp_path):
        p = tmp_path / "other.db"
        with sqlite3.connect(str(p)) as conn:
            conn.execute("CREATE TABLE foo (x INTEGER)")
        with pytest.raises(RuntimeError, match="not a JuiceFS"):
            _disjoint_counters_sqlite(str(p), 1)

    def test_rejects_missing_counter_row(self, tmp_path):
        p = _make_sqlite_index(
            tmp_path,
            counters={
                "nextInode": 1,
                # nextChunk missing
                "nextSession": 1,
            },
        )
        with pytest.raises(RuntimeError, match="missing counter 'nextChunk'"):
            _disjoint_counters_sqlite(str(p), 1)

    def test_returns_applied_offset(self, tmp_path):
        p = _make_sqlite_index(tmp_path)
        applied = _disjoint_counters_sqlite(str(p), 1_000_000)
        assert applied == 1_000_000

    def test_clamps_offset_when_near_uint64_max(self, tmp_path):
        # Build a synthetic "deep fork chain" state: one counter close to max.
        big = _COUNTER_MAX - (1 << 40)
        p = _make_sqlite_index(
            tmp_path,
            counters={
                "nextInode": 100,
                "nextChunk": big,  # the limiting counter
                "nextSession": 1,
                "usedSpace": 0,
                "totalInodes": 0,
                "nextCleanupSlices": 0,
            },
        )
        applied = _disjoint_counters_sqlite(str(p), 1 << 56)
        # Must not overflow.
        after = _read_sqlite_counters(p)
        assert after["nextChunk"] < _COUNTER_MAX
        # And must be at most half the headroom we had.
        assert applied <= (1 << 40) // 2
        # Sanity: all three counters got the same offset.
        assert after["nextChunk"] == big + applied
        assert after["nextInode"] == 100 + applied
        assert after["nextSession"] == 1 + applied

    def test_raises_when_counter_space_exhausted(self, tmp_path):
        p = _make_sqlite_index(
            tmp_path,
            counters={
                "nextInode": 1,
                "nextChunk": _COUNTER_MAX - 1000,  # almost no headroom
                "nextSession": 1,
                "usedSpace": 0,
                "totalInodes": 0,
                "nextCleanupSlices": 0,
            },
        )
        with pytest.raises(RuntimeError, match="exhausted its ID space"):
            _disjoint_counters_sqlite(str(p), 1 << 40)


@skip_no_redis
class TestDisjointCountersRedis:
    """Spin up an ephemeral redis-server with a freshly-shaped JuiceFS RDB,
    bump counters via the helper, and reload to verify.
    """

    @staticmethod
    def _build_rdb(tmp_path: Path) -> Path:
        """Produce an RDB file shaped like a JuiceFS Redis index (just the
        three counters we touch — that's all the helper validates).
        """
        port = _free_port()
        rdb = tmp_path / "dump.rdb"
        proc = subprocess.Popen(
            [
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
                str(tmp_path),
                "--dbfilename",
                rdb.name,
                "--daemonize",
                "no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("redis-server exited prematurely")
                r = _run(["redis-cli", "-p", str(port), "ping"])
                if r.returncode == 0 and "PONG" in r.stdout:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("redis-server didn't become ready")
                time.sleep(0.1)
            _run(["redis-cli", "-p", str(port), "SET", "nextchunk", "4096"])
            _run(["redis-cli", "-p", str(port), "SET", "nextinode", "1024"])
            _run(["redis-cli", "-p", str(port), "SET", "nextsession", "1"])
            _run(["redis-cli", "-p", str(port), "SAVE"])
        finally:
            _run(["redis-cli", "-p", str(port), "SHUTDOWN", "NOSAVE"])
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
        assert rdb.exists()
        return rdb

    @staticmethod
    def _read_counters(rdb: Path) -> dict:
        port = _free_port()
        proc = subprocess.Popen(
            [
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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("redis-server exited prematurely")
                r = _run(["redis-cli", "-p", str(port), "ping"])
                if r.returncode == 0 and "PONG" in r.stdout:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("redis-server didn't become ready")
                time.sleep(0.1)
            out = {}
            for k in ("nextchunk", "nextinode", "nextsession"):
                v = _run(["redis-cli", "-p", str(port), "GET", k]).stdout.strip()
                out[k] = int(v)
            return out
        finally:
            _run(["redis-cli", "-p", str(port), "SHUTDOWN", "NOSAVE"])
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)

    def test_bumps_counters(self, tmp_path):
        rdb = self._build_rdb(tmp_path)
        _disjoint_counters_redis(str(rdb), 1_000_000_000_000)
        after = self._read_counters(rdb)
        assert after == {
            "nextchunk": 4096 + 1_000_000_000_000,
            "nextinode": 1024 + 1_000_000_000_000,
            "nextsession": 1 + 1_000_000_000_000,
        }

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            _disjoint_counters_redis(str(tmp_path / "nope.rdb"), 1)

    def test_rejects_non_juicefs_rdb(self, tmp_path):
        # An RDB without the counter keys — helper must refuse.
        port = _free_port()
        rdb = tmp_path / "empty.rdb"
        proc = subprocess.Popen(
            [
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
                str(tmp_path),
                "--dbfilename",
                rdb.name,
                "--daemonize",
                "no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            time.sleep(0.5)
            _run(["redis-cli", "-p", str(port), "SET", "irrelevant", "1"])
            _run(["redis-cli", "-p", str(port), "SAVE"])
        finally:
            _run(["redis-cli", "-p", str(port), "SHUTDOWN", "NOSAVE"])
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        with pytest.raises(RuntimeError, match="missing counter"):
            _disjoint_counters_redis(str(rdb), 1)


class TestDisjointForkCounters:
    def test_dispatches_to_sqlite(self, tmp_path):
        p = _make_sqlite_index(tmp_path)
        _disjoint_fork_counters(str(p), "sqlite", 42)
        assert _read_sqlite_counters(p)["nextChunk"] == 8193 + 42

    @skip_no_redis
    def test_dispatches_to_redis(self, tmp_path):
        rdb = TestDisjointCountersRedis._build_rdb(tmp_path)
        _disjoint_fork_counters(str(rdb), "redis", 42)
        after = TestDisjointCountersRedis._read_counters(rdb)
        assert after["nextchunk"] == 4096 + 42


# ---------------------------------------------------------------------------
# 2. Unit tests — small helpers
# ---------------------------------------------------------------------------


class TestSmallHelpers:
    def test_index_filename(self):
        assert _index_filename("sqlite") == "index.db"
        assert _index_filename("redis") == "dump.rdb"
        # Unknown engine falls back to sqlite naming (defensive).
        assert _index_filename("xyz") == "index.db"

    def test_meta_url_sqlite(self, tmp_path):
        url = _meta_url(str(tmp_path), "sqlite")
        assert url.startswith("sqlite3://")
        assert url.endswith("index.db")

    def test_meta_url_redis_is_localhost(self):
        url = _meta_url("/anywhere", "redis")
        assert url == "redis://127.0.0.1:6379/0"

    def test_is_fuse_mount_false_for_random_path(self):
        # Whatever this path is, it isn't a FUSE mountpoint.
        assert _is_fuse_mount("/tmp/__definitely_not_a_mount__") is False

    def test_is_fuse_mount_handles_unreadable_mountinfo(self, monkeypatch):
        # Force open() to fail; helper must swallow OSError and return False.
        def boom(*a, **kw):
            raise OSError("no")

        monkeypatch.setattr("builtins.open", boom)
        assert _is_fuse_mount("/anything") is False

    def test_sync_filesystem_prefers_syncfs(self, tmp_path, monkeypatch):
        calls = []

        def fake_syncfs(fd):
            calls.append(("syncfs", fd))

        def fake_sync():
            calls.append(("sync",))

        monkeypatch.setattr(juicefs_mod.os, "syncfs", fake_syncfs, raising=False)
        monkeypatch.setattr(juicefs_mod.os, "sync", fake_sync)

        juicefs_mod._sync_filesystem(str(tmp_path))

        assert [call[0] for call in calls] == ["syncfs"]


class TestVolumeBackend:
    def test_default_backend_is_juicefs(self):
        backend = Volume.empty("vol-x", bucket="s3://b")._backend()

        assert backend.name == "juicefs"
        assert backend.index_filename("sqlite") == "index.db"
        assert backend.index_filename("redis") == "dump.rdb"

    def test_juicefs_mount_command_preserves_mount_tuning(self):
        backend = juicefs_mod.JuiceFSVolumeBackend()
        config = _MountConfig(
            meta_dir="/meta",
            redis_port=12345,
            cache_dir="/cache",
            writeback=True,
            upload_delay="5s",
            max_uploads=7,
            attr_cache=1.5,
            entry_cache=2.5,
            dir_entry_cache=3.5,
        )

        cmd = backend.mount_cmd(config=config, engine="redis", mount_path="/workspace")

        assert cmd[:2] == ["juicefs", "mount"]
        assert "--writeback" in cmd
        assert cmd[cmd.index("--upload-delay") + 1] == "5s"
        assert cmd[cmd.index("--max-uploads") + 1] == "7"
        assert "redis://127.0.0.1:12345/0" in cmd


class TestClientBucketUri:
    def test_s3_uri_translates_to_https(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        assert (
            _client_bucket_uri("s3://my-bucket/prefix", "s3") == "https://s3.us-west-2.amazonaws.com/my-bucket/prefix"
        )

    def test_s3_uri_falls_back_to_aws_default_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        assert _client_bucket_uri("s3://b/p", "s3") == "https://s3.eu-west-1.amazonaws.com/b/p"

    def test_s3_uri_falls_back_to_us_east_1(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        assert _client_bucket_uri("s3://b/p", "s3") == "https://s3.us-east-1.amazonaws.com/b/p"

    def test_non_s3_storage_is_pass_through(self):
        assert _client_bucket_uri("gs://my-bucket/p", "gs") == "gs://my-bucket/p"

    def test_already_translated_https_uri_is_pass_through(self):
        # If caller already passed a host-style URI, don't re-wrap it.
        u = "https://s3.us-east-1.amazonaws.com/b/p"
        assert _client_bucket_uri(u, "s3") == u


class TestDefaultBucketDerivation:
    """``_default_bucket`` derives ``{prefix}/{project}/{domain}/volumes`` from
    the active task context's raw_data_path. Tests use a fake context object
    to avoid spinning up Flyte's full runtime.
    """

    class _FakeAction:
        def __init__(self, project: str, domain: str):
            self.project = project
            self.domain = domain

    class _FakeRawDataPath:
        def __init__(self, path: str):
            self.path = path

    class _FakeTaskContext:
        def __init__(self, project: str, domain: str, path: str):
            self.action = TestDefaultBucketDerivation._FakeAction(project, domain)
            self.raw_data_path = TestDefaultBucketDerivation._FakeRawDataPath(path)

    class _FakeData:
        def __init__(self, tctx):
            self.task_context = tctx

    class _FakeCtx:
        def __init__(self, data):
            self.data = data

    def _install_ctx(self, monkeypatch, *, project: str, domain: str, raw_path: str):
        tctx = self._FakeTaskContext(project, domain, raw_path)
        ctx = self._FakeCtx(self._FakeData(tctx))
        monkeypatch.setattr("flyte.extras._volume.internal_ctx", lambda: ctx)

    def test_strips_after_project_domain_marker(self, monkeypatch):
        self._install_ctx(
            monkeypatch,
            project="proj",
            domain="dev",
            raw_path="s3://org-bucket/proj/dev/runs/run-1/raw",
        )
        assert _default_bucket() == "s3://org-bucket/proj/dev/volumes"

    def test_preserves_org_level_subprefix(self, monkeypatch):
        # Important real-world case: bucket has an org/team prefix segment.
        self._install_ctx(
            monkeypatch,
            project="proj",
            domain="dev",
            raw_path="s3://org-bucket/team-xy/proj/dev/runs/r/raw",
        )
        assert _default_bucket() == "s3://org-bucket/team-xy/proj/dev/volumes"

    def test_errors_when_no_task_context(self, monkeypatch):
        class _EmptyCtx:
            data = None

        empty = _EmptyCtx()
        monkeypatch.setattr("flyte.extras._volume.internal_ctx", lambda: empty)
        with pytest.raises(RuntimeError, match="explicit `bucket=` argument"):
            _default_bucket()

    def test_errors_when_project_or_domain_blank(self, monkeypatch):
        self._install_ctx(monkeypatch, project="", domain="dev", raw_path="s3://b/proj/dev/x")
        with pytest.raises(RuntimeError, match="missing project/domain"):
            _default_bucket()

    def test_errors_when_marker_not_in_raw_path(self, monkeypatch):
        # raw_data_path doesn't contain /proj/dev/ — derivation must fail
        # rather than silently produce a garbage prefix.
        self._install_ctx(
            monkeypatch,
            project="proj",
            domain="dev",
            raw_path="s3://b/something-else/files",
        )
        with pytest.raises(RuntimeError, match="doesn't contain"):
            _default_bucket()


class TestForkChainCounterMonotonicity:
    """Chained forks (fork-of-fork-of-fork...) must keep advancing counters
    so cousins/grandchildren don't collide either.
    """

    def test_chain_of_three_forks_strictly_monotonic(self, tmp_path):
        gen0 = _make_sqlite_index(tmp_path)
        # Snapshot each generation's counters by copying the file.
        snaps = [gen0]
        for i in range(1, 4):
            nxt = tmp_path / f"gen{i}.db"
            shutil.copyfile(str(snaps[-1]), str(nxt))
            _disjoint_counters_sqlite(str(nxt), _random_fork_offset())
            snaps.append(nxt)

        chunks = [_read_sqlite_counters(s)["nextChunk"] for s in snaps]
        # Strictly increasing (each fork bumps by at least 2**32).
        assert chunks == sorted(chunks)
        assert len(set(chunks)) == len(chunks)
        # Minimum step between adjacent generations.
        for a, b in itertools.pairwise(chunks):
            assert b - a >= (1 << 32)


# ---------------------------------------------------------------------------
# 3. Unit tests — Volume model behavior
# ---------------------------------------------------------------------------


class TestVolumeModel:
    def test_engine_defaults_to_sqlite_for_back_compat(self):
        v = Volume(name="x", bucket="s3://b")
        assert v._engine() == "sqlite"

    def test_engine_honors_metadata_engine_field(self):
        v = Volume(name="x", bucket="s3://b", metadata_engine="redis")
        assert v._engine() == "redis"

    def test_empty_requires_explicit_bucket_outside_task(self):
        # No active task context → derivation must fail with explicit message.
        with pytest.raises(RuntimeError, match="explicit `bucket=` argument"):
            Volume.empty("vol-x")

    def test_empty_sets_redis_engine_by_default(self):
        v = Volume.empty("vol-x", bucket="s3://b")
        assert v.metadata_engine == "redis"

    def test_round_trip_pydantic_serialization(self):
        from flyte.io._file import File

        v = Volume(
            name="x",
            bucket="s3://b",
            storage="s3",
            index=File(path="s3://b/idx", hash="abc"),
            metadata_engine="redis",
            used_bytes=12345,
            inode_count=42,
        )
        as_json = v.model_dump_json()
        v2 = Volume.model_validate_json(as_json)
        assert v2 == v


class TestVolumeFork:
    @pytest.mark.asyncio
    async def test_live_writeback_fork_flushes_mount_before_snapshot(self, monkeypatch):
        calls = []

        class _Proc:
            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                return 0

        async def fake_snapshot(self, *, meta_dir, tmp_prefix, flush_live=True, counter_bump=None):
            calls.append(("snapshot", meta_dir, flush_live, counter_bump))
            return File(path="s3://bucket/new-index.db"), 12

        async def fake_stats(meta_dir, engine, **kwargs):
            return 1, 2

        def fake_run_check(cmd):
            calls.append(tuple(cmd))

        async def fake_start_mount(self, *, mount_path, config, timeout):
            calls.append(("mount", mount_path, config.meta_dir, timeout))

        meta_dir = "/tmp/flyte-volume-meta"
        mount_path = "/workspace"
        monkeypatch.setattr(Volume, "_live_meta", {meta_dir})
        monkeypatch.setattr(Volume, "_live_procs", {mount_path: _Proc()})
        monkeypatch.setattr(
            Volume,
            "_live_mounts",
            {
                mount_path: _MountConfig(
                    meta_dir=meta_dir,
                    redis_port=None,
                    cache_dir="/tmp/flyte-volume-cache",
                    writeback=True,
                    upload_delay=None,
                    max_uploads=50,
                    attr_cache=60.0,
                    entry_cache=60.0,
                    dir_entry_cache=60.0,
                )
            },
        )
        monkeypatch.setattr(volume_mod, "_random_fork_offset", lambda: 42)
        monkeypatch.setattr(juicefs_mod, "_run_check", fake_run_check)
        monkeypatch.setattr(juicefs_mod, "_query_volume_stats", fake_stats)
        monkeypatch.setattr(Volume, "_snapshot_and_upload_index", fake_snapshot)
        monkeypatch.setattr(Volume, "_start_mount", fake_start_mount)

        vol = Volume(
            name="parent",
            bucket="s3://bucket/volumes",
            index=File(path="s3://bucket/old-index.db"),
            metadata_engine="sqlite",
        )

        await vol.fork("child", meta_dir=meta_dir, mount_path=mount_path)

        assert ("juicefs", "umount", "--flush", mount_path) in calls
        assert calls.index(("juicefs", "umount", "--flush", mount_path)) < calls.index(("snapshot", meta_dir, True, 42))
        assert calls.index(("snapshot", meta_dir, True, 42)) < calls.index(("mount", mount_path, meta_dir, 60.0))


class TestVolumeMount:
    @pytest.mark.asyncio
    async def test_redis_mount_uses_started_redis_port(self, tmp_path, monkeypatch):
        calls = []
        redis_port = 43123

        class _Proc:
            stdout = None

            def poll(self):
                return None

        async def fake_start_redis(meta_dir):
            calls.append(("redis", meta_dir))
            return _MetadataEngineState(proc=_Proc(), port=redis_port)

        def fake_run_check(cmd):
            calls.append(tuple(cmd))

        def fake_popen(cmd, **kwargs):
            calls.append(tuple(cmd))
            return _Proc()

        monkeypatch.setattr(Volume, "_live_meta", set())
        monkeypatch.setattr(Volume, "_live_procs", {})
        monkeypatch.setattr(Volume, "_live_mounts", {})
        monkeypatch.setattr(Volume, "_live_redis", {})
        monkeypatch.setattr(juicefs_mod, "_start_redis", fake_start_redis)
        monkeypatch.setattr(juicefs_mod, "_run_check", fake_run_check)
        monkeypatch.setattr(juicefs_mod, "_is_fuse_mount", lambda path: True)
        monkeypatch.setattr(volume_mod.subprocess, "Popen", fake_popen)

        meta_dir = str(tmp_path / "meta")
        mount_path = str(tmp_path / "mnt")
        cache_dir = str(tmp_path / "cache")
        vol = Volume.empty("redis-vol", bucket="s3://bucket/volumes", metadata_engine="redis")

        await vol.mount(meta_dir=meta_dir, mount_path=mount_path, cache_dir=cache_dir)

        expected_meta_url = f"redis://127.0.0.1:{redis_port}/0"
        assert any(call[0:2] == ("juicefs", "format") and expected_meta_url in call for call in calls)
        assert any(call[0:2] == ("juicefs", "mount") and expected_meta_url in call for call in calls)

    @pytest.mark.asyncio
    async def test_mount_failure_stops_started_redis(self, tmp_path, monkeypatch):
        calls = []

        class _Proc:
            stdout = None

            def poll(self):
                return None

        redis_state = _MetadataEngineState(proc=_Proc(), port=43124)

        async def fake_start_redis(meta_dir):
            calls.append(("redis", meta_dir))
            return redis_state

        def fake_run_check(cmd):
            calls.append(tuple(cmd))
            if cmd[0:2] == ["juicefs", "format"]:
                raise RuntimeError("format failed")

        def fake_stop_redis(state, timeout):
            calls.append(("stop-redis", state, timeout))

        monkeypatch.setattr(Volume, "_live_meta", set())
        monkeypatch.setattr(Volume, "_live_procs", {})
        monkeypatch.setattr(Volume, "_live_mounts", {})
        monkeypatch.setattr(Volume, "_live_redis", {})
        monkeypatch.setattr(juicefs_mod, "_start_redis", fake_start_redis)
        monkeypatch.setattr(juicefs_mod, "_run_check", fake_run_check)
        monkeypatch.setattr(juicefs_mod, "_stop_redis", fake_stop_redis)

        meta_dir = str(tmp_path / "meta")
        vol = Volume.empty("redis-vol", bucket="s3://bucket/volumes", metadata_engine="redis")

        with pytest.raises(RuntimeError, match="format failed"):
            await vol.mount(meta_dir=meta_dir, mount_path=str(tmp_path / "mnt"), cache_dir=str(tmp_path / "cache"))

        assert ("stop-redis", redis_state, 120.0) in calls
        assert Volume._live_redis == {}


# ---------------------------------------------------------------------------
# 4. Integration tests — real JuiceFS round-trip
# ---------------------------------------------------------------------------


class _JuiceFsLocalEnv:
    """One independent JuiceFS volume rooted in tmp_path. Uses storage=file
    and a per-volume metadata file so siblings don't interfere.
    """

    def __init__(self, tmp_path: Path, name: str, *, engine: str = "sqlite", redis_port: Optional[int] = None):
        self.tmp_path = tmp_path
        self.name = name
        self.engine = engine
        self.bucket_dir = tmp_path / "bucket"
        self.bucket_dir.mkdir(exist_ok=True)
        self.meta_dir = tmp_path / f"meta-{name}"
        self.meta_dir.mkdir(exist_ok=True)
        self.mnt_dir = tmp_path / f"mnt-{name}"
        self.mnt_dir.mkdir(exist_ok=True)
        # Per-instance disk cache. JuiceFS's default cache path is keyed by
        # volume UUID, which is identical across a parent+cold-fork pair,
        # so they would share a local cache and mask bucket-level corruption.
        # In a real Flyte pod each task gets fresh ephemeral storage, so the
        # cache never persists — we mirror that here.
        self.cache_dir = tmp_path / f"cache-{name}"
        self.cache_dir.mkdir(exist_ok=True)
        self.redis_port = redis_port  # only for engine='redis'
        self._redis_proc: Optional[subprocess.Popen] = None
        self._mount_proc: Optional[subprocess.Popen] = None

    def _meta_url(self) -> str:
        if self.engine == "redis":
            return f"redis://127.0.0.1:{self.redis_port}/0"
        return f"sqlite3://{self.meta_dir / 'index.db'}"

    def start_redis(self):
        if self.engine != "redis":
            return
        self._redis_proc = subprocess.Popen(
            [
                "redis-server",
                "--port",
                str(self.redis_port),
                "--bind",
                "127.0.0.1",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(self.meta_dir),
                "--dbfilename",
                "dump.rdb",
                "--daemonize",
                "no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10
        while True:
            r = _run(["redis-cli", "-p", str(self.redis_port), "ping"])
            if r.returncode == 0 and "PONG" in r.stdout:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("redis didn't start")
            time.sleep(0.1)

    def stop_redis(self):
        if self._redis_proc is None:
            return
        _run(["redis-cli", "-p", str(self.redis_port), "SAVE"])
        _run(["redis-cli", "-p", str(self.redis_port), "SHUTDOWN", "NOSAVE"])
        try:
            self._redis_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._redis_proc.kill()
        self._redis_proc = None

    def format(self):
        r = _run(
            [
                "juicefs",
                "format",
                "--storage",
                "file",
                "--bucket",
                str(self.bucket_dir),
                self._meta_url(),
                self.name,
            ]
        )
        if r.returncode != 0:
            raise RuntimeError(f"format failed: {r.stderr}\n{r.stdout}")

    def mount(self):
        # Force a fresh disk cache each mount so the test can't accidentally
        # cache-hit a stale chunk; see __init__ for the rationale.
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(exist_ok=True)
        self._mount_proc = subprocess.Popen(
            [
                "juicefs",
                "mount",
                "--no-syslog",
                "--cache-dir",
                str(self.cache_dir),
                self._meta_url(),
                str(self.mnt_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while True:
            if self._mount_proc.poll() is not None:
                raise RuntimeError("mount exited prematurely")
            # Use the helper under test to verify FUSE status — also exercises
            # _is_fuse_mount on a real mount.
            if _is_fuse_mount(str(self.mnt_dir)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("mount didn't come up")
            time.sleep(0.2)

    def umount(self):
        _run(["juicefs", "umount", str(self.mnt_dir)])
        if self._mount_proc is not None:
            try:
                self._mount_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._mount_proc.kill()
            self._mount_proc = None

    def close(self):
        try:
            self.umount()
        finally:
            self.stop_redis()


@skip_no_juicefs
@skip_no_fuse
class TestJuiceFsRoundTripSqlite:
    """End-to-end: format → mount → write → unmount → re-mount; data persists."""

    def test_data_persists_across_remount(self, tmp_path):
        env = _JuiceFsLocalEnv(tmp_path, "vol-one", engine="sqlite")
        env.format()
        env.mount()
        try:
            (env.mnt_dir / "hello.txt").write_text("world")
        finally:
            env.umount()
        # Remount, expect same contents
        env.mount()
        try:
            assert (env.mnt_dir / "hello.txt").read_text() == "world"
        finally:
            env.umount()


@skip_no_juicefs
@skip_no_fuse
class TestForkIsolationRegression:
    """The regression test for the chunk-key collision bug.

    Without the counter bump, parent and fork both allocated the same chunk
    slice IDs (e.g. 4097) and one side's write to the shared bucket key
    would silently overwrite the other's. This test reproduces the exact
    scenario and asserts isolation.
    """

    def _seed_parent(self, tmp_path: Path) -> _JuiceFsLocalEnv:
        env = _JuiceFsLocalEnv(tmp_path, "parent", engine="sqlite")
        env.format()
        env.mount()
        try:
            (env.mnt_dir / "seed.txt").write_text("seed")
        finally:
            env.umount()
        return env

    def test_same_size_writes_after_fork_are_isolated(self, tmp_path):
        parent = self._seed_parent(tmp_path)
        parent_db = parent.meta_dir / "index.db"

        # Cold-fork by snapshotting the metadata AND disjointing counters —
        # exactly what Volume.fork() now does on the cold path.
        fork_meta = tmp_path / "meta-fork"
        fork_meta.mkdir()
        fork_db = fork_meta / "index.db"
        _snapshot_sqlite_db(parent_db, fork_db)
        offset = _random_fork_offset()
        _disjoint_counters_sqlite(str(fork_db), offset)

        # Wire up the fork as a peer env pointing at the SAME bucket dir.
        fork_env = _JuiceFsLocalEnv(tmp_path, "fork", engine="sqlite")
        fork_env.meta_dir = fork_meta
        # Reuse parent's bucket (shared chunk space — this is the whole point).
        fork_env.bucket_dir = parent.bucket_dir

        # Both write same-size payloads with different bytes.
        parent.mount()
        try:
            (parent.mnt_dir / "new.txt").write_bytes(b"A" * 10)
        finally:
            parent.umount()

        fork_env.mount()
        try:
            (fork_env.mnt_dir / "new.txt").write_bytes(b"B" * 10)
        finally:
            fork_env.umount()

        # Each side reads its OWN value back. Without the bump, parent's
        # value would come back as "BBBBBBBBBB" (silent corruption).
        parent.mount()
        try:
            assert (parent.mnt_dir / "new.txt").read_bytes() == b"A" * 10, (
                "parent read corrupted — fork's write clobbered shared chunk key"
            )
        finally:
            parent.umount()

        fork_env.mount()
        try:
            assert (fork_env.mnt_dir / "new.txt").read_bytes() == b"B" * 10
        finally:
            fork_env.umount()

    @pytest.mark.xfail(reason="Reproduces the bug WITHOUT the bump — proves the test is sensitive to it.", strict=True)
    def test_bug_repro_without_bump_is_corrupted(self, tmp_path):
        """Sanity check: if we skip the counter bump, the bug *does* manifest.

        Marked xfail so the suite stays green; flip to a real fail if anyone
        ever reverts the fix.
        """
        parent = self._seed_parent(tmp_path)
        parent_db = parent.meta_dir / "index.db"

        fork_meta = tmp_path / "meta-fork"
        fork_meta.mkdir()
        fork_db = fork_meta / "index.db"
        _snapshot_sqlite_db(parent_db, fork_db)
        # NOTE: deliberately NOT bumping counters.

        fork_env = _JuiceFsLocalEnv(tmp_path, "fork", engine="sqlite")
        fork_env.meta_dir = fork_meta
        fork_env.bucket_dir = parent.bucket_dir

        parent.mount()
        try:
            (parent.mnt_dir / "new.txt").write_bytes(b"A" * 10)
        finally:
            parent.umount()
        fork_env.mount()
        try:
            (fork_env.mnt_dir / "new.txt").write_bytes(b"B" * 10)
        finally:
            fork_env.umount()

        parent.mount()
        try:
            assert (parent.mnt_dir / "new.txt").read_bytes() == b"A" * 10
        finally:
            parent.umount()


@skip_no_juicefs
@skip_no_redis
@skip_no_fuse
class TestMigrateMetadataEngine:
    """End-to-end migration between SQLite and Redis engines via
    ``juicefs dump | juicefs load``, exercised by spawning local engines
    and comparing namespace before/after.
    """

    @staticmethod
    def _dump_load(src_url: str, dst_url: str, tmp_path: Path):
        dump = tmp_path / "dump.json"
        r = _run(["juicefs", "dump", src_url, str(dump)])
        if r.returncode != 0:
            raise RuntimeError(f"dump failed: {r.stderr}")
        r = _run(["juicefs", "load", dst_url, str(dump)])
        if r.returncode != 0:
            raise RuntimeError(f"load failed: {r.stderr}")

    def test_sqlite_to_redis_preserves_namespace(self, tmp_path):
        # Build sqlite source with one file, then migrate to redis.
        src = _JuiceFsLocalEnv(tmp_path, "src-vol", engine="sqlite")
        src.format()
        src.mount()
        try:
            (src.mnt_dir / "hello.txt").write_text("migrate-me")
        finally:
            src.umount()

        # Migrate: dump from sqlite, load into redis (different port, same bucket).
        redis_port = _free_port()
        dst_meta = tmp_path / "dst-meta"
        dst_meta.mkdir(parents=True)
        # Start dst redis
        dst_proc = subprocess.Popen(
            [
                "redis-server",
                "--port",
                str(redis_port),
                "--bind",
                "127.0.0.1",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(dst_meta),
                "--dbfilename",
                "dump.rdb",
                "--daemonize",
                "no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                r = _run(["redis-cli", "-p", str(redis_port), "ping"])
                if r.returncode == 0 and "PONG" in r.stdout:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("dst redis didn't start")
                time.sleep(0.1)

            src_url = f"sqlite3://{src.meta_dir / 'index.db'}"
            dst_url = f"redis://127.0.0.1:{redis_port}/0"
            self._dump_load(src_url, dst_url, tmp_path)

            # Bump counters on the migrated engine to disjoint from source —
            # mirror what migrate_metadata_engine() does in production.
            offset = _random_fork_offset()
            _run(["redis-cli", "-p", str(redis_port), "INCRBY", "nextchunk", str(offset)])
            _run(["redis-cli", "-p", str(redis_port), "INCRBY", "nextinode", str(offset)])
            _run(["redis-cli", "-p", str(redis_port), "INCRBY", "nextsession", str(offset)])

            # Verify the migrated engine sees the parent's namespace by
            # pointing a new mount at the dst redis URL with the same bucket.
            dst = _JuiceFsLocalEnv(tmp_path, "src-vol", engine="redis", redis_port=redis_port)
            dst.meta_dir = dst_meta
            dst.bucket_dir = src.bucket_dir  # shared chunk space
            dst.mount()
            try:
                assert (dst.mnt_dir / "hello.txt").read_text() == "migrate-me"
                # And a new write goes to a slice ID > offset (disjoint).
                (dst.mnt_dir / "new.txt").write_text("post-migrate")
            finally:
                dst.umount()

            # Verify the bucket contains a chunk at slice >= offset (the
            # post-migrate write should have allocated from the bumped range).
            chunk_paths = sorted(p.relative_to(src.bucket_dir) for p in src.bucket_dir.rglob("*") if p.is_file())
            # Filter to chunk files: <vol>/chunks/<a>/<b>/<sliceid>_<i>_<sz>
            slice_ids = []
            for c in chunk_paths:
                parts = c.parts
                if len(parts) >= 4 and parts[1] == "chunks":
                    leaf = parts[-1]
                    try:
                        slice_ids.append(int(leaf.split("_")[0]))
                    except (ValueError, IndexError):
                        pass
            assert any(s >= offset for s in slice_ids), (
                f"migrated engine did not allocate from disjoint range; slice_ids={slice_ids}"
            )
        finally:
            _run(["redis-cli", "-p", str(redis_port), "SHUTDOWN", "NOSAVE"])
            try:
                dst_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dst_proc.kill()
