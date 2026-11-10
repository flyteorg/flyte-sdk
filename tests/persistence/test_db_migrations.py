"""Tests for flyte._persistence._db._migrate_sync."""

import sqlite3
from unittest.mock import MagicMock

import flyte._persistence._db as db


def test_migrate_skips_existing_columns_quietly(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "cache.db")
    for ddl in db._ALL_TABLE_DDLS:
        conn.execute(ddl)
    warn = MagicMock()
    monkeypatch.setattr(db.logger, "warning", warn)

    db._migrate_sync(conn)  # every column is already in the DDLs
    db._migrate_sync(conn)

    assert "storage" in [r[1] for r in conn.execute("PRAGMA table_info(bundle_cache)")]
    warn.assert_not_called()


def test_migrate_reports_a_failure_that_is_not_an_existing_column(monkeypatch):
    """A locked database is not an already-migrated one: it is logged, not passed off as done."""

    class LockedConn:
        def execute(self, stmt):
            raise sqlite3.OperationalError("database is locked")

        def commit(self):
            pass

    warn = MagicMock()
    monkeypatch.setattr(db.logger, "warning", warn)

    db._migrate_sync(LockedConn())

    assert warn.call_count == len(db._RUNS_MIGRATIONS) + len(db._BUNDLE_CACHE_MIGRATIONS)
    assert "database is locked" in warn.call_args[0][0]
