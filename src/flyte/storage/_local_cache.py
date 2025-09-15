from pathlib import Path

from flyteidl.core import interface_pb2
from sqlalchemy import Column, LargeBinary, String, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from flyte._internal.runtime import convert
from flyte._protos.workflow import run_definition_pb2

CACHE_LOCATION = "~/.flyte/local-cache"

Base = declarative_base()


class CachedOutput(Base):
    __tablename__ = "cached_outputs"

    id = Column(String, primary_key=True)
    output_literal = Column(LargeBinary, nullable=False)


class LocalTaskCache(object):
    """
    This class implements a persistent store able to cache the result of local task executions.
    """

    _engine = None
    _session_factory = None
    _initialized: bool = False
    _db_path: str = None

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
            LocalTaskCache._session_factory = sessionmaker(
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

        async with LocalTaskCache._session_factory() as session:
            await session.execute(delete(CachedOutput))
            await session.commit()

    @staticmethod
    async def get(
        task_name: str,
        inputs_hash: str,
        proto_inputs: run_definition_pb2.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
    ) -> run_definition_pb2.Outputs | None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        cache_key = convert.generate_cache_key_hash(
            task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs
        )

        async with LocalTaskCache._session_factory() as session:
            cached_output = await session.get(CachedOutput, cache_key)
            if cached_output is None:
                return None
            # Convert the cached output bytes to literal
            return run_definition_pb2.Outputs.ParseFromString(cached_output.output_literal)

    @staticmethod
    async def set(
        task_name: str,
        inputs_hash: str,
        proto_inputs: run_definition_pb2.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
        value: run_definition_pb2.Outputs,
    ) -> None:
        if not LocalTaskCache._initialized:
            await LocalTaskCache.initialize()

        cache_key = convert.generate_cache_key_hash(
            task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs
        )

        async with LocalTaskCache._session_factory() as session:
            existing = await session.get(CachedOutput, cache_key)

            # NOTE: We will directly update the value in cache if it already exists
            if existing:
                existing.output_literal = value.SerializeToString()
            else:
                new_cache = CachedOutput(id=cache_key, output_literal=value.SerializeToString())
                session.add(new_cache)

            await session.commit()
