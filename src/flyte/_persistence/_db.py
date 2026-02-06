import sqlite3
import threading
from pathlib import Path

try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

from flyte._logging import logger
from flyte.config import auto

DEFAULT_CACHE_DIR = "~/.flyte"
CACHE_LOCATION = "local-cache/cache.db"

_TASK_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS task_cache (
    key TEXT PRIMARY KEY,
    value BLOB
)
"""

_RUNS_DDL = """
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
    has_report INTEGER DEFAULT 0,
    context TEXT,
    group_name TEXT,
    log_links TEXT,
    PRIMARY KEY (run_name, action_name)
)
"""


_RUNS_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN has_report INTEGER DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN context TEXT",
    "ALTER TABLE runs ADD COLUMN group_name TEXT",
    "ALTER TABLE runs ADD COLUMN log_links TEXT",
]


def _migrate_sync(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing from an older schema."""
    for stmt in _RUNS_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


class LocalDB:
    """Shared SQLite connection manager for both task cache and run persistence."""

    _conn_sync: sqlite3.Connection | None = None
    _conn: "aiosqlite.Connection | None" = None
    _initialized: bool = False
    _lock = threading.Lock()
    _write_lock = threading.Lock()

    @staticmethod
    def _get_db_path() -> str:
        """Get the database path, creating directory if needed."""
        config = auto()
        if config.source:
            cache_dir = config.source.parent
        else:
            cache_dir = Path(DEFAULT_CACHE_DIR).expanduser()

        db_path = cache_dir / CACHE_LOCATION
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Use local DB path: {db_path}")
        return str(db_path)

    @staticmethod
    async def initialize():
        """Open async connection and create all tables."""
        with LocalDB._lock:
            if LocalDB._initialized:
                return
            if HAS_AIOSQLITE:
                await LocalDB._initialize_async()
            else:
                LocalDB._initialize_sync_inner()

    @staticmethod
    async def _initialize_async():
        db_path = LocalDB._get_db_path()
        conn = await aiosqlite.connect(db_path)
        await conn.execute(_TASK_CACHE_DDL)
        await conn.execute(_RUNS_DDL)
        await conn.commit()
        LocalDB._conn = conn
        # Also open a sync connection for sync callers
        sync_conn = sqlite3.connect(db_path, check_same_thread=False)
        sync_conn.execute(_TASK_CACHE_DDL)
        sync_conn.execute(_RUNS_DDL)
        _migrate_sync(sync_conn)
        LocalDB._conn_sync = sync_conn
        LocalDB._initialized = True

    @staticmethod
    def initialize_sync():
        """Open sync connection and create all tables."""
        with LocalDB._lock:
            if LocalDB._initialized:
                return
            LocalDB._initialize_sync_inner()

    @staticmethod
    def _initialize_sync_inner():
        db_path = LocalDB._get_db_path()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(_TASK_CACHE_DDL)
        conn.execute(_RUNS_DDL)
        _migrate_sync(conn)
        LocalDB._conn_sync = conn
        LocalDB._initialized = True

    @staticmethod
    def get_sync() -> sqlite3.Connection:
        """Get sync connection, auto-initializing if needed."""
        if not LocalDB._initialized:
            LocalDB.initialize_sync()
        if LocalDB._conn_sync is None:
            raise RuntimeError("LocalDB not properly initialized (sync)")
        return LocalDB._conn_sync

    @staticmethod
    async def get_async() -> "aiosqlite.Connection":
        """Get async connection, auto-initializing if needed."""
        if not LocalDB._initialized:
            await LocalDB.initialize()
        if LocalDB._conn is None:
            raise RuntimeError("LocalDB not properly initialized (async)")
        return LocalDB._conn

    @staticmethod
    async def close():
        """Close all connections."""
        with LocalDB._lock:
            if LocalDB._conn:
                await LocalDB._conn.close()
                LocalDB._conn = None
            if LocalDB._conn_sync:
                LocalDB._conn_sync.close()
                LocalDB._conn_sync = None
            LocalDB._initialized = False

    @staticmethod
    def close_sync():
        """Close sync connection."""
        with LocalDB._lock:
            if LocalDB._conn_sync:
                LocalDB._conn_sync.close()
                LocalDB._conn_sync = None
            if LocalDB._conn is None:
                LocalDB._initialized = False
