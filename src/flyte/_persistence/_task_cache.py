from flyteidl2.task import common_pb2

from flyte._internal.runtime import convert
from flyte._persistence._db import HAS_AIOSQLITE, LocalDB


class LocalTaskCache(object):
    """Persistent store for caching local task execution results.

    Uses the shared ``LocalDB`` connection manager.
    """

    @staticmethod
    async def initialize():
        """Initialize the cache (delegates to LocalDB)."""
        await LocalDB.initialize()

    @staticmethod
    async def clear():
        """Clear all cache entries."""
        if HAS_AIOSQLITE:
            conn = await LocalDB.get_async()
            await conn.execute("DELETE FROM task_cache")
            await conn.commit()
        else:
            conn = LocalDB.get_sync()
            conn.execute("DELETE FROM task_cache")
            conn.commit()

    @staticmethod
    async def get(cache_key: str) -> convert.Outputs | None:
        if HAS_AIOSQLITE:
            return await LocalTaskCache._get_async(cache_key)
        else:
            return LocalTaskCache._get_sync(cache_key)

    @staticmethod
    async def _get_async(cache_key: str) -> convert.Outputs | None:
        conn = await LocalDB.get_async()
        async with conn.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                outputs_bytes = row[0]
                outputs = common_pb2.Outputs()
                outputs.ParseFromString(outputs_bytes)
                return convert.Outputs(proto_outputs=outputs)
        return None

    @staticmethod
    def _get_sync(cache_key: str) -> convert.Outputs | None:
        conn = LocalDB.get_sync()
        cursor = conn.execute("SELECT value FROM task_cache WHERE key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            outputs_bytes = row[0]
            outputs = common_pb2.Outputs()
            outputs.ParseFromString(outputs_bytes)
            return convert.Outputs(proto_outputs=outputs)
        return None

    @staticmethod
    async def set(cache_key: str, value: convert.Outputs) -> None:
        if HAS_AIOSQLITE:
            await LocalTaskCache._set_async(cache_key, value)
        else:
            LocalTaskCache._set_sync(cache_key, value)

    @staticmethod
    async def _set_async(cache_key: str, value: convert.Outputs) -> None:
        conn = await LocalDB.get_async()
        output_bytes = value.proto_outputs.SerializeToString()
        await conn.execute("INSERT OR REPLACE INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes))
        await conn.commit()

    @staticmethod
    def _set_sync(cache_key: str, value: convert.Outputs) -> None:
        conn = LocalDB.get_sync()
        output_bytes = value.proto_outputs.SerializeToString()
        conn.execute("INSERT OR REPLACE INTO task_cache (key, value) VALUES (?, ?)", (cache_key, output_bytes))
        conn.commit()

    @staticmethod
    async def close():
        """Close the database connection."""
        await LocalDB.close()

    @staticmethod
    def close_sync():
        """Close the sync database connection."""
        LocalDB.close_sync()
