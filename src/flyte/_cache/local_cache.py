from pathlib import Path

from flyteidl.core import interface_pb2
from sqlalchemy import Column, LargeBinary, String, delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from flyte._internal.runtime import convert
from flyte._protos.workflow import run_definition_pb2

CACHE_LOCATION = "~/.flyte/local-cache"


class Base(DeclarativeBase):
    pass


class CachedOutput(Base):
    __tablename__ = "cached_outputs"

    id = Column(String, primary_key=True)
    output_bytes = Column(LargeBinary, nullable=False)


class LocalTaskCache(object):
    """
    This class implements a persistent store able to cache the result of local task executions.
    """

    _engine: AsyncEngine | None = None
    _session_factory: async_sessionmaker[AsyncSession] | None = None
    _initialized: bool = False
    _db_path: str | None = None

    @staticmethod
    def _get_cache_path() -> str:
        """Get the cache database path, creating directory if needed."""
        if LocalTaskCache._db_path is None:
            cache_dir = Path(CACHE_LOCATION).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            LocalTaskCache._db_path = str(cache_dir / "cache.db")
        return LocalTaskCache._db_path

    @staticmethod
    async def initialize():
        """Initialize the cache with database connection."""
        if not LocalTaskCache._initialized:
            db_path = LocalTaskCache._get_cache_path()
            LocalTaskCache._engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
            LocalTaskCache._session_factory = async_sessionmaker(
                bind=LocalTaskCache._engine, class_=AsyncSession, expire_on_commit=False
            )

            async with LocalTaskCache._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            LocalTaskCache._initialized = True

    @staticmethod
    async def clear():
        """Clear all cache entries."""
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        async with LocalTaskCache._session_factory() as session:  # type: ignore[misc]
            await session.execute(delete(CachedOutput))
            await session.commit()

    @staticmethod
    async def get(
        task_name: str,
        inputs_hash: str,
        proto_inputs: convert.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
    ) -> convert.Outputs | None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        cache_key = convert.generate_cache_key_hash(
            task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs.proto_inputs
        )

        async with LocalTaskCache._session_factory() as session:  # type: ignore[misc]
            cached_output = await session.get(CachedOutput, cache_key)
            if cached_output is None:
                return None
            # Convert the cached output bytes to literal
            outputs = run_definition_pb2.Outputs()
            outputs.ParseFromString(cached_output.output_bytes)  # type: ignore[arg-type]
            return convert.Outputs(proto_outputs=outputs)

    @staticmethod
    async def set(
        task_name: str,
        inputs_hash: str,
        proto_inputs: convert.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
        value: convert.Outputs,
    ) -> None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        cache_key = convert.generate_cache_key_hash(
            task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs.proto_inputs
        )

        async with LocalTaskCache._session_factory() as session:  # type: ignore[misc]
            existing = await session.get(CachedOutput, cache_key)

            # NOTE: We will directly update the value in cache if it already exists
            if existing:
                existing.output_bytes = value.proto_outputs.SerializeToString()  # type: ignore[assignment]
            else:
                new_cache = CachedOutput(id=cache_key, output_bytes=value.proto_outputs.SerializeToString())
                session.add(new_cache)

            await session.commit()
