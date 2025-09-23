from pathlib import Path

import aiosqlite

from flyte._internal.runtime import convert
from flyte._logging import logger
from flyte._protos.workflow import run_definition_pb2
from flyte.config import auto

DEFAILT_CACHE_DIR = "~/.flyte"
CACHE_LOCATION = "local-cache/cache.db"


class LocalTaskCache(object):
    """
    This class implements a persistent store able to cache the result of local task executions.
    """

    _conn: aiosqlite.Connection | None = None
    _initialized: bool = False

    @staticmethod
    def _get_cache_path() -> str:
        """Get the cache database path, creating directory if needed."""
        config = auto()
        if config.source:
            cache_dir = config.source.parent
        else:
            cache_dir = Path(DEFAILT_CACHE_DIR).expanduser()

        cache_path = cache_dir / CACHE_LOCATION
        # Ensure the directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Use local cache path: {cache_path}")
        return str(cache_path)

    @staticmethod
    async def initialize():
        """Initialize the cache with database connection."""
        if not LocalTaskCache._initialized:
            db_path = LocalTaskCache._get_cache_path()
            conn = await aiosqlite.connect(db_path)

            # Create the task_cache table if it doesn't exist
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
    async def clear():
        """Clear all cache entries."""
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")
        await LocalTaskCache._conn.execute("DELETE FROM task_cache")
        await LocalTaskCache._conn.commit()

    @staticmethod
    async def get(cache_key: str) -> convert.Outputs | None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        async with LocalTaskCache._conn.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                outputs_bytes = row[0]
                outputs = run_definition_pb2.Outputs()
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
        if LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        # Check if cache entry already exists
        existing = await LocalTaskCache.get(cache_key)

        output_bytes = value.proto_outputs.SerializeToString()

        # NOTE: We will directly update the value in cache if it already exists
        if existing:
            await LocalTaskCache._conn.execute(
                "UPDATE task_cache SET value = ? WHERE key = ?", (output_bytes, cache_key)
            )
        else:
            await LocalTaskCache._conn.execute(
                "INSERT INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes)
            )

        await LocalTaskCache._conn.commit()
