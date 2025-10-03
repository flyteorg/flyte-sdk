import sqlite3
from pathlib import Path

try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

from flyteidl2.task import common_pb2

from flyte._internal.runtime import convert
from flyte._logging import logger
from flyte.config import auto

DEFAULT_CACHE_DIR = "~/.flyte"
CACHE_LOCATION = "local-cache/cache.db"


class LocalTaskCache(object):
    """
    This class implements a persistent store able to cache the result of local task executions.
    """

    _conn: "aiosqlite.Connection | None" = None
    _conn_sync: sqlite3.Connection | None = None
    _initialized: bool = False

    @staticmethod
    def _get_cache_path() -> str:
        """Get the cache database path, creating directory if needed."""
        config = auto()
        if config.source:
            cache_dir = config.source.parent
        else:
            cache_dir = Path(DEFAULT_CACHE_DIR).expanduser()

        cache_path = cache_dir / CACHE_LOCATION
        # Ensure the directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Use local cache path: {cache_path}")
        return str(cache_path)

    @staticmethod
    async def initialize():
        """Initialize the cache with database connection."""
        if not LocalTaskCache._initialized:
            if HAS_AIOSQLITE:
                await LocalTaskCache._initialize_async()
            else:
                LocalTaskCache._initialize_sync()

    @staticmethod
    async def _initialize_async():
        """Initialize async cache connection."""
        db_path = LocalTaskCache._get_cache_path()
        conn = await aiosqlite.connect(db_path)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_cache (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)
        await conn.commit()
        LocalTaskCache._conn = conn
        LocalTaskCache._initialized = True

    @staticmethod
    def _initialize_sync():
        """Initialize sync cache connection."""
        db_path = LocalTaskCache._get_cache_path()
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_cache (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)
        conn.commit()
        LocalTaskCache._conn_sync = conn
        LocalTaskCache._initialized = True

    @staticmethod
    async def clear():
        """Clear all cache entries."""
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        if HAS_AIOSQLITE:
            await LocalTaskCache._clear_async()
        else:
            LocalTaskCache._clear_sync()

    @staticmethod
    async def _clear_async():
        """Clear all cache entries (async)."""
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")
        await LocalTaskCache._conn.execute("DELETE FROM task_cache")
        await LocalTaskCache._conn.commit()

    @staticmethod
    def _clear_sync():
        """Clear all cache entries (sync)."""
        if LocalTaskCache._conn_sync is None:
            raise RuntimeError("Cache not properly initialized")
        LocalTaskCache._conn_sync.execute("DELETE FROM task_cache")
        LocalTaskCache._conn_sync.commit()

    @staticmethod
    async def get(cache_key: str) -> convert.Outputs | None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        if HAS_AIOSQLITE:
            return await LocalTaskCache._get_async(cache_key)
        else:
            return LocalTaskCache._get_sync(cache_key)

    @staticmethod
    async def _get_async(cache_key: str) -> convert.Outputs | None:
        """Get cache entry (async)."""
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        async with LocalTaskCache._conn.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                outputs_bytes = row[0]
                outputs = common_pb2.Outputs()
                outputs.ParseFromString(outputs_bytes)
                return convert.Outputs(proto_outputs=outputs)
        return None

    @staticmethod
    def _get_sync(cache_key: str) -> convert.Outputs | None:
        """Get cache entry (sync)."""
        if LocalTaskCache._conn_sync is None:
            raise RuntimeError("Cache not properly initialized")

        cursor = LocalTaskCache._conn_sync.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            outputs_bytes = row[0]
            outputs = common_pb2.Outputs()
            outputs.ParseFromString(outputs_bytes)
            return convert.Outputs(proto_outputs=outputs)
        return None

    @staticmethod
    async def set(
        cache_key: str,
        value: convert.Outputs,
    ) -> None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        if HAS_AIOSQLITE:
            await LocalTaskCache._set_async(cache_key, value)
        else:
            LocalTaskCache._set_sync(cache_key, value)

    @staticmethod
    async def _set_async(
        cache_key: str,
        value: convert.Outputs,
    ) -> None:
        """Set cache entry (async)."""
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        output_bytes = value.proto_outputs.SerializeToString()
        await LocalTaskCache._conn.execute(
            "INSERT OR REPLACE INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes)
        )
        await LocalTaskCache._conn.commit()

    @staticmethod
    def _set_sync(
        cache_key: str,
        value: convert.Outputs,
    ) -> None:
        """Set cache entry (sync)."""
        if LocalTaskCache._conn_sync is None:
            raise RuntimeError("Cache not properly initialized")

        output_bytes = value.proto_outputs.SerializeToString()
        LocalTaskCache._conn_sync.execute(
            "INSERT OR REPLACE INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes)
        )
        LocalTaskCache._conn_sync.commit()

    @staticmethod
    async def close():
        """Close the database connection."""
        if HAS_AIOSQLITE:
            await LocalTaskCache._close_async()
        else:
            LocalTaskCache._close_sync()

    @staticmethod
    async def _close_async():
        """Close async database connection."""
        if LocalTaskCache._conn:
            await LocalTaskCache._conn.close()
            LocalTaskCache._conn = None
        LocalTaskCache._initialized = False

    @staticmethod
    def _close_sync():
        """Close sync database connection."""
        if LocalTaskCache._conn_sync:
            LocalTaskCache._conn_sync.close()
            LocalTaskCache._conn_sync = None
        LocalTaskCache._initialized = False
