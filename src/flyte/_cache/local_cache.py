import sqlite3
from pathlib import Path

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

    _conn: sqlite3.Connection | None = None
    _cursor: sqlite3.Cursor | None = None
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
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create the task_cache table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_cache (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)
            conn.commit()

            LocalTaskCache._conn = conn
            LocalTaskCache._cursor = cursor
            LocalTaskCache._initialized = True

    @staticmethod
    async def clear():
        """Clear all cache entries."""
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()
        if LocalTaskCache._cursor is None or LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")
        LocalTaskCache._cursor.execute("DELETE FROM task_cache")
        LocalTaskCache._conn.commit()

    @staticmethod
    async def get(cache_key: str) -> convert.Outputs | None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()
        if LocalTaskCache._cursor is None or LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        LocalTaskCache._cursor.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,))
        row = LocalTaskCache._cursor.fetchone()

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
        if LocalTaskCache._cursor is None or LocalTaskCache._conn is None:
            raise RuntimeError("Cache not properly initialized")

        # Check if cache entry already exists
        existing = await LocalTaskCache.get(cache_key)

        output_bytes = value.proto_outputs.SerializeToString()

        # NOTE: We will directly update the value in cache if it already exists
        if existing:
            LocalTaskCache._cursor.execute("UPDATE task_cache SET value = ? WHERE key = ?", (output_bytes, cache_key))
        else:
            LocalTaskCache._cursor.execute(
                "INSERT INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes)
            )

        LocalTaskCache._conn.commit()
