import sqlite3
import threading

import pytest

from flyte._persistence._db import HAS_AIOSQLITE, LocalDB


@pytest.fixture(autouse=True)
def _reset_db(tmp_path, monkeypatch):
    """Reset LocalDB state before each test and point it at a temp directory."""
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False
    # Monkeypatch _get_db_path to use tmp_path
    db_path = str(tmp_path / "cache.db")
    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(lambda: db_path))
    yield
    if LocalDB._conn_sync:
        LocalDB._conn_sync.close()
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False


def test_initialize_sync_creates_tables():
    LocalDB.initialize_sync()
    assert LocalDB._initialized is True
    conn = LocalDB.get_sync()

    # Check task_cache table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_cache'")
    assert cursor.fetchone() is not None

    # Check runs table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    assert cursor.fetchone() is not None


def test_initialize_sync_is_idempotent():
    LocalDB.initialize_sync()
    conn1 = LocalDB.get_sync()
    LocalDB.initialize_sync()
    conn2 = LocalDB.get_sync()
    assert conn1 is conn2


def test_get_sync_auto_initializes():
    assert LocalDB._initialized is False
    conn = LocalDB.get_sync()
    assert conn is not None
    assert LocalDB._initialized is True


def test_close_sync():
    LocalDB.initialize_sync()
    assert LocalDB._initialized is True
    LocalDB.close_sync()
    assert LocalDB._conn_sync is None


def test_thread_safety():
    """Multiple threads can get the sync connection without error."""
    errors = []

    def worker():
        try:
            conn = LocalDB.get_sync()
            conn.execute("SELECT 1 FROM task_cache LIMIT 1")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")
async def test_initialize_async():
    await LocalDB.initialize()
    assert LocalDB._initialized is True
    conn = await LocalDB.get_async()
    assert conn is not None


@pytest.mark.asyncio
async def test_initialize_async_fallback_sync():
    """When aiosqlite is not available, async initialize falls through to sync."""
    await LocalDB.initialize()
    assert LocalDB._initialized is True
    # Sync connection should always be available
    conn = LocalDB.get_sync()
    assert conn is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")
async def test_close_async():
    await LocalDB.initialize()
    await LocalDB.close()
    assert LocalDB._initialized is False
    assert LocalDB._conn is None


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")
async def test_get_async_after_sync_init():
    """Regression: a sync-only init (e.g. run persistence warming the DB) must
    not leave `get_async()` unable to ever open the async connection.

    This is the Google Colab failure mode: `flyte.init(local_persistence=True)`
    sync-initializes the DB at run start, and a later cached child task's
    `LocalTaskCache.get` raised `RuntimeError: LocalDB not properly initialized
    (async)` because init was gated on the shared `_initialized` flag.
    """
    LocalDB.initialize_sync()
    assert LocalDB._initialized is True
    assert LocalDB._conn is None

    conn = await LocalDB.get_async()
    assert conn is not None
    assert LocalDB._conn is conn
    # The pre-existing sync connection is preserved, not replaced.
    assert LocalDB._conn_sync is not None
    # And the async connection is usable.
    async with conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_cache'") as cursor:
        assert await cursor.fetchone() is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOSQLITE, reason="aiosqlite not installed")
async def test_get_sync_after_async_init_reuses_connection():
    """Async-first init opens both connections; later sync callers reuse them."""
    await LocalDB.initialize()
    sync_conn_before = LocalDB._conn_sync
    conn = LocalDB.get_sync()
    assert conn is sync_conn_before
    # Re-initializing async must not replace the existing sync connection.
    await LocalDB.initialize()
    assert LocalDB._conn_sync is sync_conn_before


def test_migration_adds_missing_columns(tmp_path, monkeypatch):
    """Simulate an old DB without the new columns and verify migration adds them."""
    db_path = str(tmp_path / "old.db")
    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(lambda: db_path))
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False

    # Create old-schema table without the 4 new columns
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_name TEXT NOT NULL,
            action_name TEXT NOT NULL,
            task_name TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            inputs TEXT,
            outputs TEXT,
            error TEXT,
            start_time REAL,
            end_time REAL,
            parent_id TEXT,
            short_name TEXT,
            output_path TEXT,
            cache_enabled INTEGER DEFAULT 0,
            cache_hit INTEGER DEFAULT 0,
            PRIMARY KEY (run_name, action_name)
        )
    """)
    conn.execute("INSERT INTO runs (run_name, action_name, task_name, status) VALUES ('r1', 'a0', 't1', 'running')")
    conn.commit()
    conn.close()

    # Now initialize — migration should add the missing columns
    LocalDB.initialize_sync()
    c = LocalDB.get_sync()

    # Verify we can insert with the new columns
    c.execute(
        "UPDATE runs SET has_report=1, context='{}', group_name='g', log_links='[]' WHERE run_name='r1'",
    )
    c.commit()

    cursor = c.execute("SELECT has_report, context, group_name, log_links FROM runs WHERE run_name='r1'")
    row = cursor.fetchone()
    assert row == (1, "{}", "g", "[]")
