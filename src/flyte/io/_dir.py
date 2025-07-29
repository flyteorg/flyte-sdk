from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

from flyteidl.core import literals_pb2, types_pb2
from fsspec.asyn import AsyncFileSystem
from mashumaro.types import SerializableType
from pydantic import BaseModel, model_validator

import flyte.storage as storage
from flyte.io._file import File
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError

# Type variable for the directory format
T = TypeVar("T")


class Dir(BaseModel, Generic[T], SerializableType):
    """
    A generic directory class representing a directory with files of a specified format.
    Provides both async and sync interfaces for directory operations.
    Users are responsible for handling all I/O - the type transformer for Dir does not do any automatic uploading
    or downloading of files.

    The generic type T represents the format of the files in the directory.

    Example:
        ```python
        # Async usage
        from pandas import DataFrame
        data_dir = Dir[DataFrame](path="s3://my-bucket/data/")

        # Walk through files
        async for file in data_dir.walk():
            async with file.open() as f:
                content = await f.read()

        # Sync alternative
        for file in data_dir.walk_sync():
            with file.open_sync() as f:
                content = f.read()
        ```
    """

    # Represents either a local or remote path.
    path: str
    name: Optional[str] = None
    format: str = ""

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def pre_init(cls, data):
        if data.get("name") is None:
            data["name"] = Path(data["path"]).name
        return data

    def _serialize(self) -> Dict[str, Optional[str]]:
        pyd_dump = self.model_dump()
        return pyd_dump

    @classmethod
    def _deserialize(cls, file_dump: Dict[str, Optional[str]]) -> Dir:
        return cls.model_validate(file_dump)

    @classmethod
    def schema_match(cls, incoming: dict):
        this_schema = cls.model_json_schema()
        current_required = this_schema.get("required")
        incoming_required = incoming.get("required")
        if (
            current_required
            and incoming_required
            and incoming.get("type") == this_schema.get("type")
            and incoming.get("title") == this_schema.get("title")
            and set(current_required) == set(incoming_required)
        ):
            return True

    async def walk(self, recursive: bool = True, max_depth: Optional[int] = None) -> AsyncIterator[File[T]]:
        """
        Asynchronously walk through the directory and yield File objects.

        Args:
            recursive: If True, recursively walk subdirectories
            max_depth: Maximum depth for recursive walking

        Yields:
            File objects for each file found in the directory

        Example:
            ```python
            async for file in directory.walk():
                local_path = await file.download()
                # Process the file
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        if recursive is False:
            max_depth = 2

        # Note if the path is actually just a file, no walking is done.
        if isinstance(fs, AsyncFileSystem):
            async for parent, _, files in fs._walk(self.path, maxdepth=max_depth):
                for file in files:
                    full_file = fs.unstrip_protocol(parent + fs.sep + file)
                    yield File[T](path=full_file)
        else:
            for parent, _, files in fs.walk(self.path, maxdepth=max_depth):
                for file in files:
                    if "file" in fs.protocol:
                        full_file = os.path.join(parent, file)
                    else:
                        full_file = fs.unstrip_protocol(parent + fs.sep + file)
                    yield File[T](path=full_file)

    def walk_sync(
        self, recursive: bool = True, file_pattern: str = "*", max_depth: Optional[int] = None
    ) -> Iterator[File[T]]:
        """
        Synchronously walk through the directory and yield File objects.

        Args:
            recursive: If True, recursively walk subdirectories
            file_pattern: Glob pattern to filter files
            max_depth: Maximum depth for recursive walking

        Yields:
            File objects for each file found in the directory

        Example:
            ```python
            for file in directory.walk_sync():
                local_path = file.download_sync()
                # Process the file
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        for parent, _, files in fs.walk(self.path, maxdepth=max_depth):
            for file in files:
                if "file" in fs.protocol:
                    full_file = os.path.join(parent, file)
                else:
                    full_file = fs.unstrip_protocol(parent + fs.sep + file)
                yield File[T](path=full_file)

    async def list_files(self) -> List[File[T]]:
        """
        Asynchronously get a list of all files in the directory (non-recursive).

        Returns:
            A list of File objects

        Example:
            ```python
            files = await directory.list_files()
            for file in files:
                # Process the file
            ```
        """
        # todo: this should probably also just defer to fsspec.find()
        files = []
        async for file in self.walk(recursive=False):
            files.append(file)
        return files

    def list_files_sync(self) -> List[File[T]]:
        """
        Synchronously get a list of all files in the directory (non-recursive).

        Returns:
            A list of File objects

        Example:
            ```python
            files = directory.list_files_sync()
            for file in files:
                # Process the file
            ```
        """
        return list(self.walk_sync(recursive=False))

    async def download(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Asynchronously download the entire directory to a local path.

        Args:
            local_path: The local path to download the directory to. If None, a temporary
                       directory will be used.

        Returns:
            The path to the downloaded directory

        Example:
            ```python
            local_dir = await directory.download('/tmp/my_data/')
            ```
        """
        local_dest = str(local_path) if local_path else str(storage.get_random_local_path())
        if not storage.is_remote(self.path):
            if not local_path or local_path == self.path:
                # Skip copying
                return self.path
            else:
                # Shell out to a thread to copy
                import asyncio
                import shutil

                async def copy_tree():
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: shutil.copytree(self.path, local_dest, dirs_exist_ok=True))

                await copy_tree()
        return await storage.get(self.path, local_dest, recursive=True)

    def download_sync(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Synchronously download the entire directory to a local path.

        Args:
            local_path: The local path to download the directory to. If None, a temporary
                       directory will be used.

        Returns:
            The path to the downloaded directory

        Example:
            ```python
            local_dir = directory.download_sync('/tmp/my_data/')
            ```
        """
        local_dest = str(local_path) if local_path else str(storage.get_random_local_path())
        if not storage.is_remote(self.path):
            if not local_path or local_path == self.path:
                # Skip copying
                return self.path
            else:
                # Shell out to a thread to copy
                import shutil

                shutil.copytree(self.path, local_dest, dirs_exist_ok=True)

        # Figure this out when we figure out the final sync story
        raise NotImplementedError("Sync download is not implemented for remote paths")

    @classmethod
    async def from_local(cls, local_path: Union[str, Path], remote_path: Optional[str] = None) -> Dir[T]:
        """
        Asynchronously create a new Dir by uploading a local directory to the configured remote store.

        Args:
            local_path: Path to the local directory
            remote_path: Optional path to store the directory remotely. If None, a path will be generated.

        Returns:
            A new Dir instance pointing to the uploaded directory

        Example:
            ```python
            remote_dir = await Dir[DataFrame].from_local('/tmp/data_dir/', 's3://bucket/data/')
            ```
        """
        local_path_str = str(local_path)
        dirname = os.path.basename(os.path.normpath(local_path_str))

        output_path = await storage.put(from_path=local_path_str, to_path=remote_path, recursive=True)
        return cls(path=output_path, name=dirname)

    @classmethod
    def from_local_sync(cls, local_path: Union[str, Path], remote_path: Optional[str] = None) -> Dir[T]:
        """
        Synchronously create a new Dir by uploading a local directory to the configured remote store.

        Args:
            local_path: Path to the local directory
            remote_path: Optional path to store the directory remotely. If None, a path will be generated.

        Returns:
            A new Dir instance pointing to the uploaded directory

        Example:
            ```python
            remote_dir = Dir[DataFrame].from_local_sync('/tmp/data_dir/', 's3://bucket/data/')
            ```
        """
        # Implement this after we figure out the final sync story
        raise NotImplementedError("Sync upload is not implemented for remote paths")

    async def exists(self) -> bool:
        """
        Asynchronously check if the directory exists.

        Returns:
            True if the directory exists, False otherwise

        Example:
            ```python
            if await directory.exists():
                # Process the directory
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        if isinstance(fs, AsyncFileSystem):
            return await fs._exists(self.path)
        else:
            return fs.exists(self.path)

    def exists_sync(self) -> bool:
        """
        Synchronously check if the directory exists.

        Returns:
            True if the directory exists, False otherwise

        Example:
            ```python
            if directory.exists_sync():
                # Process the directory
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        return fs.exists(self.path)

    async def get_file(self, file_name: str) -> Optional[File[T]]:
        """
        Asynchronously get a specific file from the directory.

        Args:
            file_name: The name of the file to get

        Returns:
            A File instance if the file exists, None otherwise

        Example:
            ```python
            file = await directory.get_file("data.csv")
            if file:
                # Process the file
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        file_path = fs.sep.join([self.path, file_name])
        file = File[T](path=file_path)

        if fs.exists(file_path):
            return file
        return None

    def get_file_sync(self, file_name: str) -> Optional[File[T]]:
        """
        Synchronously get a specific file from the directory.

        Args:
            file_name: The name of the file to get

        Returns:
            A File instance if the file exists, None otherwise

        Example:
            ```python
            file = directory.get_file_sync("data.csv")
            if file:
                # Process the file
            ```
        """
        file_path = os.path.join(self.path, file_name)
        file = File[T](path=file_path)

        if file.exists_sync():
            return file
        return None


class DirTransformer(TypeTransformer[Dir]):
    """
    Transformer for Dir objects. This type transformer does not handle any i/o. That is now the responsibility of the
    user.
    """

    def __init__(self):
        super().__init__(name="Dir", t=Dir)

    def get_literal_type(self, t: Type[Dir]) -> types_pb2.LiteralType:
        """Get the Flyte literal type for a File type."""
        return types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                # todo: set format from generic
                format="",  # Format is determined by the generic type T
                dimensionality=types_pb2.BlobType.BlobDimensionality.MULTIPART,
            )
        )

    async def to_literal(
        self,
        python_val: Dir,
        python_type: Type[Dir],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        """Convert a Dir object to a Flyte literal."""
        if not isinstance(python_val, Dir):
            raise TypeTransformerFailedError(f"Expected Dir object, received {type(python_val)}")

        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                blob=literals_pb2.Blob(
                    metadata=literals_pb2.BlobMetadata(
                        type=types_pb2.BlobType(
                            format=python_val.format, dimensionality=types_pb2.BlobType.BlobDimensionality.MULTIPART
                        )
                    ),
                    uri=python_val.path,
                )
            )
        )

    async def to_python_value(
        self,
        lv: literals_pb2.Literal,
        expected_python_type: Type[Dir],
    ) -> Dir:
        """Convert a Flyte literal to a File object."""
        if not lv.scalar.HasField("blob"):
            raise TypeTransformerFailedError(f"Expected blob literal, received {lv}")
        if not lv.scalar.blob.metadata.type.dimensionality == types_pb2.BlobType.BlobDimensionality.MULTIPART:
            raise TypeTransformerFailedError(
                f"Expected multipart, received {lv.scalar.blob.metadata.type.dimensionality}"
            )

        uri = lv.scalar.blob.uri
        filename = Path(uri).name
        f: Dir = Dir(path=uri, name=filename, format=lv.scalar.blob.metadata.type.format)
        return f

    def guess_python_type(self, literal_type: types_pb2.LiteralType) -> Type[Dir]:
        """Guess the Python type from a Flyte literal type."""
        if (
            literal_type.HasField("blob")
            and literal_type.blob.dimensionality == types_pb2.BlobType.BlobDimensionality.MULTIPART
        ):
            return Dir
        raise ValueError(f"Cannot guess python type from {literal_type}")


TypeEngine.register(DirTransformer())
