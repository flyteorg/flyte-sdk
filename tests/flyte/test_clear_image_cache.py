import pytest

from flyte._internal.imagebuild.image_builder import clear_image_cache
from flyte._persistence._db import LocalDB


@pytest.fixture(autouse=True)
def _reset_db(tmp_path, monkeypatch):
    """Reset LocalDB state before each test and point it at a temp directory."""
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False
    db_path = str(tmp_path / "cache.db")
    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(lambda: db_path))
    yield
    if LocalDB._conn_sync:
        LocalDB._conn_sync.close()
    LocalDB._conn = None
    LocalDB._conn_sync = None
    LocalDB._initialized = False


def _insert(key: str, image_uri: str) -> None:
    conn = LocalDB.get_sync()
    with LocalDB._write_lock:
        conn.execute(
            "INSERT OR REPLACE INTO image_cache (key, image_uri, created_at) VALUES (?, ?, ?)",
            (key, image_uri, 0.0),
        )
        conn.commit()


def _count() -> int:
    return LocalDB.get_sync().execute("SELECT COUNT(*) FROM image_cache").fetchone()[0]


def test_clear_image_cache_removes_all_entries():
    _insert("k1", "repo:tag1")
    _insert("k2", "repo:tag2")
    assert _count() == 2

    removed = clear_image_cache()

    assert removed == 2
    assert _count() == 0


def test_clear_image_cache_empty_is_noop():
    assert _count() == 0
    removed = clear_image_cache()
    assert removed == 0
    assert _count() == 0
