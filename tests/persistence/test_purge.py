from pathlib import Path

import pytest

from flyte._persistence._db import LocalDB


@pytest.fixture(autouse=True)
def _reset_db(tmp_path, monkeypatch):
    """Reset LocalDB state before each test and point it at a temp cache dir."""
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False
    cache_dir = tmp_path / "local-cache"

    def _db_path() -> str:
        db = cache_dir / "cache.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        return str(db)

    # The root conftest autouse-patches _get_db_path; override both here so the
    # DB and the cache dir we purge stay consistent.
    monkeypatch.setattr(LocalDB, "_get_cache_dir", staticmethod(lambda: cache_dir))
    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(_db_path))
    yield
    if LocalDB._conn_sync:
        LocalDB._conn_sync.close()
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False


def test_purge_removes_entire_cache_dir():
    LocalDB.initialize_sync()
    cache_dir = Path(LocalDB._get_cache_dir())
    assert cache_dir.exists()
    assert (cache_dir / "cache.db").exists()

    returned = LocalDB.purge()

    assert returned == cache_dir
    assert not cache_dir.exists()
    # State is reset so the next access re-initializes cleanly.
    assert LocalDB._initialized is False
    assert LocalDB._conn_sync is None


def test_purge_missing_dir_is_noop():
    cache_dir = Path(LocalDB._get_cache_dir())
    assert not cache_dir.exists()

    returned = LocalDB.purge()

    assert returned == cache_dir
    assert not cache_dir.exists()


def test_purge_then_reinitialize_recreates_db():
    LocalDB.initialize_sync()
    LocalDB.purge()

    # Re-initializing after a purge should recreate the directory and DB.
    LocalDB.initialize_sync()
    cache_dir = Path(LocalDB._get_cache_dir())
    assert cache_dir.exists()
    assert (cache_dir / "cache.db").exists()
