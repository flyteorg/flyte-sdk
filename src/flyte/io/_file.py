from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import (
    IO,
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)

import aiofiles
from flyteidl.core import literals_pb2, types_pb2
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from mashumaro.types import SerializableType
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

import flyte.storage as storage
from flyte._context import internal_ctx
from flyte._initialize import requires_initialization
from flyte._logging import logger
from flyte.io._hashing_io import AsyncHashingReader, HashingWriter, HashMethod, PrecomputedValue
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError

# Type variable for the file format
T = TypeVar("T")


class File(BaseModel, Generic[T], SerializableType):
    """
    A generic file class representing a file with a specified format.
    Provides both async and sync interfaces for file operations.
    Users must handle all I/O operations themselves by instantiating this class with the appropriate class methods.

    The generic type T represents the format of the file.

    Example:
        ```python
        # Async usage
        from pandas import DataFrame
        csv_file = File[DataFrame](path="s3://my-bucket/data.csv")

        async with csv_file.open() as f:
            content = await f.read()

        # Sync alternative
        with csv_file.open_sync() as f:
            content = f.read()
        ```

    Example: Read a file input in a Task.
    ```
    @env.task
    async def my_task(file: File[DataFrame]):
        async with file.open() as f:
            df = pd.read_csv(f)
    ```

    Example: Write a file by streaming it directly to blob storage
    ```
    @env.task
    async def my_task() -> File[DataFrame]:
        df = pd.DataFrame(...)
        file = File.new_remote()
        async with file.open("wb") as f:
            df.to_csv(f)
        # No additional uploading will be done here.
        return file
    ```
    Example: Write a file by writing it locally first, and then uploading it.
    ```
    @env.task
    async def my_task() -> File[DataFrame]:
        # write to /tmp/data.csv
        return File.from_local("/tmp/data.csv", optional="s3://my-bucket/data.csv")
    ```

    Example: From an existing remote file
    ```
    @env.task
    async def my_task() -> File[DataFrame]:
        return File.from_existing_remote("s3://my-bucket/data.csv")
    ```

    Example: Take a remote file as input and return the same one, should not do any copy
    ```
    @env.task
    async def my_task(file: File[DataFrame]) -> File[DataFrame]:
        return file
    ```

    Args:
        path: The path to the file (can be local or remote)
        name: Optional name for the file (defaults to basename of path)
    """

    path: str
    name: Optional[str] = None
    format: str = ""
    hash: Optional[str] = None
    hash_method: Annotated[Optional[HashMethod], Field(default=None, exclude=True), SkipJsonSchema()]

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
    def _deserialize(cls, file_dump: Dict[str, Optional[str]]) -> File:
        return File.model_validate(file_dump)

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

    @classmethod
    @requires_initialization
    def new_remote(cls, hash_method: Optional[HashMethod] = None) -> File[T]:
        """
        Create a new File reference for a remote file that will be written to.

        Example:
        ```
        @env.task
        async def my_task() -> File[DataFrame]:
            df = pd.DataFrame(...)
            file = File.new_remote()
            async with file.open("wb") as f:
                df.to_csv(f)
            return file
        ```
        """
        ctx = internal_ctx()

        return cls(path=ctx.raw_data.get_random_remote_path(), hash_method=hash_method)

    @classmethod
    def from_existing_remote(cls, remote_path: str, known_hash_value: Optional[str] = None) -> File[T]:
        """
        Create a File reference from an existing remote file.

        Example:
        ```python
        @env.task
        async def my_task() -> File[DataFrame]:
            return File.from_existing_remote("s3://my-bucket/data.csv")
        ```

        Args:
            remote_path: The remote path to the existing file
            known_hash_value: Optional hash value to use for discovery purposes. If not specified, the value of this
              File object will be hashed (basically the path, not the contents).
        """
        return cls(path=remote_path, hash=known_hash_value)

    @asynccontextmanager
    async def open(
        self,
        mode: str = "rb",
        block_size: Optional[int] = None,
        cache_type: str = "readahead",
        cache_options: Optional[dict] = None,
        compression: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[IO[Any]]:
        """
        Asynchronously open the file and return a file-like object.

        Args:
            mode: The mode to open the file in (default: 'rb')
            block_size: Size of blocks for reading (bytes)
            cache_type: Caching mechanism to use ('readahead', 'mmap', 'bytes', 'none')
            cache_options: Dictionary of options for the cache
            compression: Compression format or None for auto-detection
            **kwargs: Additional arguments passed to fsspec's open method

        Returns:
            An async file-like object

        Example:
            ```python
            async with file.open('rb') as f:
                data = await f.read()
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)

        # Set up cache options if provided
        if cache_options is None:
            cache_options = {}

        # Configure the open parameters
        open_kwargs = {"mode": mode, **kwargs}
        if compression:
            open_kwargs["compression"] = compression

        if block_size:
            open_kwargs["block_size"] = block_size

        # Apply caching strategy
        if cache_type != "none":
            open_kwargs["cache_type"] = cache_type
            open_kwargs["cache_options"] = cache_options

        # Use aiofiles for local files
        if fs.protocol == "file":
            async with aiofiles.open(self.path, mode=mode, **kwargs) as f:
                yield f
        else:
            # This code is broadly similar to what storage.get_stream does, but without actually reading from the stream
            file_handle = None
            try:
                if "b" not in mode:
                    raise ValueError("Mode must include 'b' for binary access, when using remote files.")
                if isinstance(fs, AsyncFileSystem):
                    file_handle = await fs.open_async(self.path, mode)
                    yield file_handle
                    return
            except NotImplementedError:
                logger.debug(f"{fs} doesn't implement 'open_async', falling back to sync")
            finally:
                if file_handle is not None:
                    file_handle.close()

            with fs.open(self.path, mode) as file_handle:
                if self.hash_method and self.hash is None:
                    fh = HashingWriter(file_handle, accumulator=self.hash_method)
                    yield fh
                    self.hash = fh.result()
                    fh.close()
                else:
                    yield file_handle
                    file_handle.close()

    def exists_sync(self) -> bool:
        """
        Synchronously check if the file exists.

        Returns:
            True if the file exists, False otherwise

        Example:
            ```python
            if file.exists_sync():
                # Process the file
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        return fs.exists(self.path)

    @contextmanager
    def open_sync(
        self,
        mode: str = "rb",
        block_size: Optional[int] = None,
        cache_type: str = "readahead",
        cache_options: Optional[dict] = None,
        compression: Optional[str] = None,
        **kwargs,
    ) -> Generator[IO[Any]]:
        """
        Synchronously open the file and return a file-like object.

        Args:
            mode: The mode to open the file in (default: 'rb')
            block_size: Size of blocks for reading (bytes)
            cache_type: Caching mechanism to use ('readahead', 'mmap', 'bytes', 'none')
            cache_options: Dictionary of options for the cache
            compression: Compression format or None for auto-detection
            **kwargs: Additional arguments passed to fsspec's open method

        Returns:
            A file-like object

        Example:
            ```python
            with file.open_sync('rb') as f:
                data = f.read()
            ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)

        # Set up cache options if provided
        if cache_options is None:
            cache_options = {}

        # Configure the open parameters
        open_kwargs = {"mode": mode, "compression": compression, **kwargs}

        if block_size:
            open_kwargs["block_size"] = block_size

        # Apply caching strategy
        if cache_type != "none":
            open_kwargs["cache_type"] = cache_type
            open_kwargs["cache_options"] = cache_options

        with fs.open(self.path, **open_kwargs) as f:
            yield f

    # TODO sync needs to be implemented
    async def download(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Asynchronously download the file to a local path.

        Args:
            local_path: The local path to download the file to. If None, a temporary
                       directory will be used.

        Returns:
            The path to the downloaded file

        Example:
            ```python
            local_file = await file.download('/tmp/myfile.csv')
            ```
        """
        if local_path is None:
            local_path = storage.get_random_local_path(file_path_or_file_name=local_path)
        else:
            local_path = str(Path(local_path).absolute())

        fs = storage.get_underlying_filesystem(path=self.path)

        # If it's already a local file, just copy it
        if "file" in fs.protocol:
            # Use aiofiles for async copy
            async with aiofiles.open(self.path, "rb") as src:
                async with aiofiles.open(local_path, "wb") as dst:
                    await dst.write(await src.read())
            return str(local_path)

        # Otherwise download from remote using async functionality
        await storage.get(self.path, str(local_path))
        return str(local_path)

    @classmethod
    @requires_initialization
    async def from_local(
        cls,
        local_path: Union[str, Path],
        remote_destination: Optional[str] = None,
        hash_method: Optional[HashMethod] = None,
    ) -> File[T]:
        """
        Create a new File object from a local file that will be uploaded to the configured remote store.

        Args:
            local_path: Path to the local file
            remote_destination: Optional path to store the file remotely. If None, a path will be generated.
            hash_method: Optional HashMethod to use for determining a task's cache key if this File object is used as
              an input to said task. If not specified, the cache key will just be computed based on this object's
              attributes (i.e. path, name, format, etc.).
              If there is a set value you want to use, please pass an instance of the PrecomputedValue HashMethod.

        Returns:
            A new File instance pointing to the uploaded file

        Example:
            ```python
            remote_file = await File[DataFrame].from_local('/tmp/data.csv', 's3://bucket/data.csv')
            ```
        """
        if not os.path.exists(local_path):
            raise ValueError(f"File not found: {local_path}")

        remote_path = remote_destination or internal_ctx().raw_data.get_random_remote_path()
        protocol = get_protocol(remote_path)
        filename = Path(local_path).name

        # If remote_destination was not set by the user, and the configured raw data path is also local,
        # then let's optimize by not uploading.
        hash_value = None
        if "file" in protocol:
            if remote_destination is None:
                path = str(Path(local_path).absolute())
            else:
                # Otherwise, actually make a copy of the file
                async with aiofiles.open(local_path, "rb") as src:
                    async with aiofiles.open(remote_path, "wb") as dst:
                        if hash_method:
                            dst_wrapper = HashingWriter(dst, accumulator=hash_method)
                            await dst_wrapper.write(await src.read())
                            hash_value = dst_wrapper.result()
                        else:
                            await dst.write(await src.read())
                path = str(Path(remote_path).absolute())
        else:
            # Otherwise upload to remote using async storage layer
            if hash_method:
                # We can skip the wrapper if the hash method is just a precomputed value
                if not isinstance(hash_method, PrecomputedValue):
                    async with aiofiles.open(local_path, "rb") as src:
                        src_wrapper = AsyncHashingReader(src, accumulator=hash_method)
                        path = await storage.put_stream(src_wrapper, to_path=remote_path)
                        hash_value = src_wrapper.result()
                else:
                    path = await storage.put(str(local_path), remote_path)
                    hash_value = hash_method.result()
            else:
                path = await storage.put(str(local_path), remote_path)

        f = cls(path=path, name=filename, hash_method=hash_method, hash=hash_value)
        return f


class FileTransformer(TypeTransformer[File]):
    """
    Transformer for File objects. This type transformer does not handle any i/o. That is now the responsibility of the
    user.
    """

    def __init__(self):
        super().__init__(name="File", t=File)

    def get_literal_type(self, t: Type[File]) -> types_pb2.LiteralType:
        """Get the Flyte literal type for a File type."""
        return types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                # todo: set format from generic
                format="",  # Format is determined by the generic type T
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

    async def to_literal(
        self,
        python_val: File,
        python_type: Type[File],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        """Convert a File object to a Flyte literal."""
        if not isinstance(python_val, File):
            raise TypeTransformerFailedError(f"Expected File object, received {type(python_val)}")

        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                blob=literals_pb2.Blob(
                    metadata=literals_pb2.BlobMetadata(
                        type=types_pb2.BlobType(
                            format=python_val.format, dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE
                        )
                    ),
                    uri=python_val.path,
                )
            ),
            hash=python_val.hash if python_val.hash else None,
        )

    async def to_python_value(
        self,
        lv: literals_pb2.Literal,
        expected_python_type: Type[File],
    ) -> File:
        """Convert a Flyte literal to a File object."""
        if not lv.scalar.HasField("blob"):
            raise TypeTransformerFailedError(f"Expected blob literal, received {lv}")
        if not lv.scalar.blob.metadata.type.dimensionality == types_pb2.BlobType.BlobDimensionality.SINGLE:
            raise TypeTransformerFailedError(
                f"Expected single part blob, received {lv.scalar.blob.metadata.type.dimensionality}"
            )

        uri = lv.scalar.blob.uri
        filename = Path(uri).name
        hash_value = lv.hash if lv.hash else None
        f: File = File(path=uri, name=filename, format=lv.scalar.blob.metadata.type.format, hash=hash_value)
        return f

    def guess_python_type(self, literal_type: types_pb2.LiteralType) -> Type[File]:
        """Guess the Python type from a Flyte literal type."""
        if (
            literal_type.HasField("blob")
            and literal_type.blob.dimensionality == types_pb2.BlobType.BlobDimensionality.SINGLE
            and literal_type.blob.format != "PythonPickle"  # see pickle transformer
        ):
            return File
        raise ValueError(f"Cannot guess python type from {literal_type}")


TypeEngine.register(FileTransformer())
