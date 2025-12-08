from __future__ import annotations

import inspect
import os
import typing
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
from flyteidl2.core import literals_pb2, types_pb2
from fsspec.utils import get_protocol
from mashumaro.types import SerializableType
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

import flyte.errors
import flyte.storage as storage
from flyte._context import internal_ctx
from flyte._initialize import requires_initialization
from flyte.io._hashing_io import AsyncHashingReader, HashingWriter, HashMethod, PrecomputedValue
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError

if typing.TYPE_CHECKING:
    from obstore import AsyncReadableFile, AsyncWritableFile

if typing.TYPE_CHECKING:
    from obstore import AsyncReadableFile, AsyncWritableFile

# Type variable for the file format
T = TypeVar("T")


class File(BaseModel, Generic[T], SerializableType):
    """
    A generic file class representing a file with a specified format.
    Provides both async and sync interfaces for file operations. All methods without _sync suffix are async.

    The class should be instantiated using one of the class methods. The constructor should be used only to
    instantiate references to existing remote objects.

    The generic type T represents the format of the file.

    Important methods:
    - `from_existing_remote`: Create a File object from an existing remote file.
    - `new_remote`: Create a new File reference for a remote file that will be written to.

    **Asynchronous methods**:
    - `open`: Asynchronously open the file and return a file-like object.
    - `download`: Asynchronously download the file to a local path.
    - `from_local`: Asynchronously create a File object from a local file, uploading it to remote storage.
    - `exists`: Asynchronously check if the file exists.

    **Synchronous methods** (suffixed with `_sync`):
    - `open_sync`: Synchronously open the file and return a file-like object.
    - `download_sync`: Synchronously download the file to a local path.
    - `from_local_sync`: Synchronously create a File object from a local file, uploading it to remote storage.
    - `exists_sync`: Synchronously check if the file exists.

    Example: Read a file input in a Task (Async).

    ```python
    @env.task
    async def read_file(file: File) -> str:
        async with file.open("rb") as f:
            content = bytes(await f.read())
            return content.decode("utf-8")
    ```

    Example: Read a file input in a Task (Sync).

    ```python
    @env.task
    def read_file_sync(file: File) -> str:
        with file.open_sync("rb") as f:
            content = f.read()
            return content.decode("utf-8")
    ```

    Example: Write a file by streaming it directly to blob storage (Async).

    ```python
    @env.task
    async def write_file() -> File:
        file = File.new_remote()
        async with file.open("wb") as f:
            await f.write(b"Hello, World!")
        return file
    ```

    Example: Upload a local file to remote storage (Async).

    ```python
    @env.task
    async def upload_file() -> File:
        # Write to local file first
        with open("/tmp/data.csv", "w") as f:
            f.write("col1,col2\\n1,2\\n3,4\\n")
        # Upload to remote storage
        return await File.from_local("/tmp/data.csv")
    ```

    Example: Upload a local file to remote storage (Sync).

    ```python
    @env.task
    def upload_file_sync() -> File:
        # Write to local file first
        with open("/tmp/data.csv", "w") as f:
            f.write("col1,col2\\n1,2\\n3,4\\n")
        # Upload to remote storage
        return File.from_local_sync("/tmp/data.csv")
    ```

    Example: Download a file to local storage (Async).

    ```python
    @env.task
    async def download_file(file: File) -> str:
        local_path = await file.download()
        # Process the local file
        with open(local_path, "r") as f:
            return f.read()
    ```

    Example: Download a file to local storage (Sync).

    ```python
    @env.task
    def download_file_sync(file: File) -> str:
        local_path = file.download_sync()
        # Process the local file
        with open(local_path, "r") as f:
            return f.read()
    ```

    Example: Reference an existing remote file.

    ```python
    @env.task
    async def process_existing_file() -> str:
        file = File.from_existing_remote("s3://my-bucket/data.csv")
        async with file.open("rb") as f:
            content = await f.read()
            return content.decode("utf-8")
    ```

    Example: Check if a file exists (Async).

    ```python
    @env.task
    async def check_file(file: File) -> bool:
        return await file.exists()
    ```

    Example: Check if a file exists (Sync).

    ```python
    @env.task
    def check_file_sync(file: File) -> bool:
        return file.exists_sync()
    ```

    Example: Pass through a file without copying.

    ```python
    @env.task
    async def pass_through(file: File) -> File:
        # No copy occurs - just passes the reference
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
    hash_method: Annotated[Optional[HashMethod], Field(default=None, exclude=True), SkipJsonSchema()] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def pre_init(cls, data):
        """Internal: Pydantic validator to set default name from path. Not intended for direct use."""
        if data.get("name") is None:
            data["name"] = Path(data["path"]).name
        return data

    def _serialize(self) -> Dict[str, Optional[str]]:
        """Internal: Serialize File to dictionary. Not intended for direct use."""
        pyd_dump = self.model_dump()
        return pyd_dump

    @classmethod
    def _deserialize(cls, file_dump: Dict[str, Optional[str]]) -> File:
        """Internal: Deserialize File from dictionary. Not intended for direct use."""
        return File.model_validate(file_dump)

    @classmethod
    def schema_match(cls, incoming: dict):
        """Internal: Check if incoming schema matches File schema. Not intended for direct use."""
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
    def new_remote(cls, file_name: Optional[str] = None, hash_method: Optional[HashMethod | str] = None) -> File[T]:
        """
        Create a new File reference for a remote file that will be written to.

        Use this when you want to create a new file and write to it directly without creating a local file first.

        Example (Async):

        ```python
        @env.task
        async def create_csv() -> File:
            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            file = File.new_remote()
            async with file.open("wb") as f:
                df.to_csv(f)
            return file
        ```

        Args:
            file_name: Optional string specifying a remote file name. If not set,
                      a generated file name will be returned.
            hash_method: Optional HashMethod or string to use for cache key computation. If a string is provided,
                        it will be used as a precomputed cache key. If a HashMethod is provided, it will be used
                        to compute the hash as data is written.

        Returns:
            A new File instance with a generated remote path
        """
        ctx = internal_ctx()
        known_cache_key = hash_method if isinstance(hash_method, str) else None
        method = hash_method if isinstance(hash_method, HashMethod) else None

        return cls(
            path=ctx.raw_data.get_random_remote_path(file_name=file_name), hash=known_cache_key, hash_method=method
        )

    @classmethod
    def from_existing_remote(cls, remote_path: str, file_cache_key: Optional[str] = None) -> File[T]:
        """
        Create a File reference from an existing remote file.

        Use this when you want to reference a file that already exists in remote storage without uploading it.

        Example:

        ```python
        @env.task
        async def process_existing_file() -> str:
            file = File.from_existing_remote("s3://my-bucket/data.csv")
            async with file.open("rb") as f:
                content = await f.read()
            return content.decode("utf-8")
        ```

        Args:
            remote_path: The remote path to the existing file
            file_cache_key: Optional hash value to use for cache key computation. If not specified, the cache key
                           will be computed based on the file's attributes (path, name, format).

        Returns:
            A new File instance pointing to the existing remote file
        """
        return cls(path=remote_path, hash=file_cache_key)

    @asynccontextmanager
    async def open(
        self,
        mode: str = "rb",
        block_size: Optional[int] = None,
        cache_type: str = "readahead",
        cache_options: Optional[dict] = None,
        compression: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[AsyncWritableFile, AsyncReadableFile, "HashingWriter"], None]:
        """
        Asynchronously open the file and return a file-like object.

        Use this method in async tasks to read from or write to files directly.

        Example (Async Read):

        ```python
        @env.task
        async def read_file(f: File) -> str:
            async with f.open("rb") as fh:
                content = bytes(await fh.read())
                return content.decode("utf-8")
        ```

        Example (Async Write):

        ```python
        @env.task
        async def write_file() -> File:
            f = File.new_remote()
            async with f.open("wb") as fh:
                await fh.write(b"Hello, World!")
            return f
        ```

        Example (Streaming Read):

        ```python
        @env.task
        async def stream_read(f: File) -> str:
            content_parts = []
            async with f.open("rb", block_size=1024) as fh:
                while True:
                    chunk = await fh.read()
                    if not chunk:
                        break
                    content_parts.append(chunk)
            return b"".join(content_parts).decode("utf-8")
        ```

        Args:
            mode: The mode to open the file in (default: 'rb'). Common modes: 'rb' (read binary),
                  'wb' (write binary), 'rt' (read text), 'wt' (write text)
            block_size: Size of blocks for reading in bytes. Useful for streaming large files.
            cache_type: Caching mechanism to use ('readahead', 'mmap', 'bytes', 'none')
            cache_options: Dictionary of options for the cache
            compression: Compression format or None for auto-detection
            **kwargs: Additional arguments passed to fsspec's open method

        Returns:
            An async file-like object that can be used with async read/write operations
        """
        # Check if we should use obstore bypass
        try:
            fh = await storage.open(
                self.path,
                mode=mode,
                cache_type=cache_type,
                cache_options=cache_options,
                compression=compression,
                block_size=block_size,
                **kwargs,
            )
            try:
                yield fh
                return
            finally:
                if inspect.iscoroutinefunction(fh.close):
                    await fh.close()
                else:
                    fh.close()
        except flyte.errors.OnlyAsyncIOSupportedError:
            # Fall back to aiofiles
            fs = storage.get_underlying_filesystem(path=self.path)
            if "file" in fs.protocol:
                async with aiofiles.open(self.path, mode=mode, **kwargs) as f:
                    yield f
                return
            raise

    async def exists(self) -> bool:
        """
        Asynchronously check if the file exists.

        Example (Async):

        ```python
        @env.task
        async def check_file(f: File) -> bool:
            if await f.exists():
                print("File exists!")
                return True
            return False
        ```

        Returns:
            True if the file exists, False otherwise
        """
        return await storage.exists(self.path)

    def exists_sync(self) -> bool:
        """
        Synchronously check if the file exists.

        Use this in non-async tasks or when you need synchronous file existence checking.

        Example (Sync):

        ```python
        @env.task
        def check_file_sync(f: File) -> bool:
            if f.exists_sync():
                print("File exists!")
                return True
            return False
        ```

        Returns:
            True if the file exists, False otherwise
        """
        return storage.exists_sync(self.path)

    @contextmanager
    def open_sync(
        self,
        mode: str = "rb",
        block_size: Optional[int] = None,
        cache_type: str = "readahead",
        cache_options: Optional[dict] = None,
        compression: Optional[str] = None,
        **kwargs,
    ) -> Generator[IO[Any], None, None]:
        """
        Synchronously open the file and return a file-like object.

        Use this method in non-async tasks to read from or write to files directly.

        Example (Sync Read):

        ```python
        @env.task
        def read_file_sync(f: File) -> str:
            with f.open_sync("rb") as fh:
                content = fh.read()
                return content.decode("utf-8")
        ```

        Example (Sync Write):

        ```python
        @env.task
        def write_file_sync() -> File:
            f = File.new_remote()
            with f.open_sync("wb") as fh:
                fh.write(b"Hello, World!")
            return f
        ```

        Args:
            mode: The mode to open the file in (default: 'rb'). Common modes: 'rb' (read binary),
                  'wb' (write binary), 'rt' (read text), 'wt' (write text)
            block_size: Size of blocks for reading in bytes. Useful for streaming large files.
            cache_type: Caching mechanism to use ('readahead', 'mmap', 'bytes', 'none')
            cache_options: Dictionary of options for the cache
            compression: Compression format or None for auto-detection
            **kwargs: Additional arguments passed to fsspec's open method

        Returns:
            A file-like object that can be used with standard read/write operations
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

        Use this when you need to download a remote file to your local filesystem for processing.

        Example (Async):

        ```python
        @env.task
        async def download_and_process(f: File) -> str:
            local_path = await f.download()
            # Now process the local file
            with open(local_path, "r") as fh:
                return fh.read()
        ```

        Example (Download to specific path):

        ```python
        @env.task
        async def download_to_path(f: File) -> str:
            local_path = await f.download("/tmp/myfile.csv")
            return local_path
        ```

        Args:
            local_path: The local path to download the file to. If None, a temporary
                       directory will be used and a path will be generated.

        Returns:
            The absolute path to the downloaded file
        """
        if local_path is None:
            local_path = storage.get_random_local_path(file_path_or_file_name=self.path)
        else:
            # Preserve trailing separator if present (Path.absolute() strips it)
            local_path_str = str(local_path)
            has_trailing_sep = local_path_str.endswith(os.sep)
            local_path = str(Path(local_path).absolute())
            if has_trailing_sep:
                local_path = local_path + os.sep

        fs = storage.get_underlying_filesystem(path=self.path)

        # If it's already a local file, just copy it
        if "file" in fs.protocol:
            # Apply directory logic for local-to-local copies
            local_path_for_copy = local_path
            if isinstance(local_path, str):
                local_path_obj = Path(local_path)
                # Check if it's a directory or ends with separator
                if local_path.endswith(os.sep) or (local_path_obj.exists() and local_path_obj.is_dir()):
                    remote_filename = Path(self.path).name
                    local_path_for_copy = str(local_path_obj / remote_filename)

            # Ensure parent directory exists
            Path(local_path_for_copy).parent.mkdir(parents=True, exist_ok=True)

            # Use aiofiles for async copy
            async with aiofiles.open(self.path, "rb") as src:
                async with aiofiles.open(local_path_for_copy, "wb") as dst:
                    await dst.write(await src.read())
            return str(local_path_for_copy)

        # Otherwise download from remote using async functionality
        result_path = await storage.get(self.path, str(local_path))
        return result_path

    def download_sync(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Synchronously download the file to a local path.

        Use this in non-async tasks when you need to download a remote file to your local filesystem.

        Example (Sync):

        ```python
        @env.task
        def download_and_process_sync(f: File) -> str:
            local_path = f.download_sync()
            # Now process the local file
            with open(local_path, "r") as fh:
                return fh.read()
        ```

        Example (Download to specific path):

        ```python
        @env.task
        def download_to_path_sync(f: File) -> str:
            local_path = f.download_sync("/tmp/myfile.csv")
            return local_path
        ```

        Args:
            local_path: The local path to download the file to. If None, a temporary
                       directory will be used and a path will be generated.

        Returns:
            The absolute path to the downloaded file
        """
        if local_path is None:
            local_path = storage.get_random_local_path(file_path_or_file_name=self.path)
        else:
            # Preserve trailing separator if present (Path.absolute() strips it)
            local_path_str = str(local_path)
            has_trailing_sep = local_path_str.endswith(os.sep)
            local_path = str(Path(local_path).absolute())
            if has_trailing_sep:
                local_path = local_path + os.sep

        fs = storage.get_underlying_filesystem(path=self.path)

        # If it's already a local file, just copy it
        if "file" in fs.protocol:
            # Apply directory logic for local-to-local copies
            local_path_for_copy = local_path
            if isinstance(local_path, str):
                local_path_obj = Path(local_path)
                # Check if it's a directory or ends with separator
                if local_path.endswith(os.sep) or (local_path_obj.exists() and local_path_obj.is_dir()):
                    remote_filename = Path(self.path).name
                    local_path_for_copy = str(local_path_obj / remote_filename)

            # Ensure parent directory exists
            Path(local_path_for_copy).parent.mkdir(parents=True, exist_ok=True)

            # Use standard file operations for sync copy
            import shutil

            shutil.copy2(self.path, local_path_for_copy)
            return str(local_path_for_copy)

        # Otherwise download from remote using sync functionality
        # Use the sync version of storage operations
        with fs.open(self.path, "rb") as src:
            with open(local_path, "wb") as dst:
                dst.write(src.read())
        return str(local_path)

    @classmethod
    @requires_initialization
    def from_local_sync(
        cls,
        local_path: Union[str, Path],
        remote_destination: Optional[str] = None,
        hash_method: Optional[HashMethod | str] = None,
    ) -> File[T]:
        """
        Synchronously create a new File object from a local file by uploading it to remote storage.

        Use this in non-async tasks when you have a local file that needs to be uploaded to remote storage.

        Example (Sync):

        ```python
        @env.task
        def upload_local_file_sync() -> File:
            # Create a local file
            with open("/tmp/data.csv", "w") as f:
                f.write("col1,col2\n1,2\n3,4\n")

            # Upload to remote storage
            remote_file = File.from_local_sync("/tmp/data.csv")
            return remote_file
        ```

        Example (With specific destination):

        ```python
        @env.task
        def upload_to_specific_path() -> File:
            remote_file = File.from_local_sync("/tmp/data.csv", "s3://my-bucket/data.csv")
            return remote_file
        ```

        Args:
            local_path: Path to the local file
            remote_destination: Optional remote path to store the file. If None, a path will be automatically generated.
            hash_method: Optional HashMethod or string to use for cache key computation. If a string is provided,
                        it will be used as a precomputed cache key. If a HashMethod is provided, it will compute
                        the hash during upload. If not specified, the cache key will be based on file attributes.

        Returns:
            A new File instance pointing to the uploaded remote file
        """
        if not os.path.exists(local_path):
            raise ValueError(f"File not found: {local_path}")

        remote_path = remote_destination or internal_ctx().raw_data.get_random_remote_path()
        protocol = get_protocol(remote_path)
        filename = Path(local_path).name

        # If remote_destination was not set by the user, and the configured raw data path is also local,
        # then let's optimize by not uploading.
        hash_value = hash_method if isinstance(hash_method, str) else None
        hash_method_obj = hash_method if isinstance(hash_method, HashMethod) else None

        if "file" in protocol:
            if remote_destination is None:
                path = str(Path(local_path).absolute())
            else:
                # Otherwise, actually make a copy of the file
                import shutil

                if hash_method_obj:
                    # For hash computation, we need to read and write manually
                    with open(local_path, "rb") as src:
                        with open(remote_path, "wb") as dst:
                            dst_wrapper = HashingWriter(dst, accumulator=hash_method_obj)
                            dst_wrapper.write(src.read())
                            hash_value = dst_wrapper.result()
                            dst_wrapper.close()
                else:
                    shutil.copy2(local_path, remote_path)
                path = str(Path(remote_path).absolute())
        else:
            # Otherwise upload to remote using sync storage layer
            fs = storage.get_underlying_filesystem(path=remote_path)

            if hash_method_obj:
                # We can skip the wrapper if the hash method is just a precomputed value
                if not isinstance(hash_method_obj, PrecomputedValue):
                    with open(local_path, "rb") as src:
                        # For sync operations, we need to compute hash manually
                        data = src.read()
                        hash_method_obj.update(memoryview(data))
                        hash_value = hash_method_obj.result()

                    # Now write the data to remote
                    with fs.open(remote_path, "wb") as dst:
                        dst.write(data)
                    path = remote_path
                else:
                    # Use sync file operations
                    with open(local_path, "rb") as src:
                        with fs.open(remote_path, "wb") as dst:
                            dst.write(src.read())
                    path = remote_path
                    hash_value = hash_method_obj.result()
            else:
                # Simple sync copy
                with open(local_path, "rb") as src:
                    with fs.open(remote_path, "wb") as dst:
                        dst.write(src.read())
                path = remote_path

        f = cls(path=path, name=filename, hash_method=hash_method_obj, hash=hash_value)
        return f

    @classmethod
    @requires_initialization
    async def from_local(
        cls,
        local_path: Union[str, Path],
        remote_destination: Optional[str] = None,
        hash_method: Optional[HashMethod | str] = None,
    ) -> File[T]:
        """
        Asynchronously create a new File object from a local file by uploading it to remote storage.

        Use this in async tasks when you have a local file that needs to be uploaded to remote storage.

        Example (Async):

        ```python
        @env.task
        async def upload_local_file() -> File:
            # Create a local file
            async with aiofiles.open("/tmp/data.csv", "w") as f:
                await f.write("col1,col2\n1,2\n3,4\n")

            # Upload to remote storage
            remote_file = await File.from_local("/tmp/data.csv")
            return remote_file
        ```

        Example (With specific destination):

        ```python
        @env.task
        async def upload_to_specific_path() -> File:
            remote_file = await File.from_local("/tmp/data.csv", "s3://my-bucket/data.csv")
            return remote_file
        ```

        Args:
            local_path: Path to the local file
            remote_destination: Optional remote path to store the file. If None, a path will be automatically generated.
            hash_method: Optional HashMethod or string to use for cache key computation. If a string is provided,
                        it will be used as a precomputed cache key. If a HashMethod is provided, it will compute
                        the hash during upload. If not specified, the cache key will be based on file attributes.

        Returns:
            A new File instance pointing to the uploaded remote file
        """
        if not os.path.exists(local_path):
            raise ValueError(f"File not found: {local_path}")

        filename = Path(local_path).name
        remote_path = remote_destination or internal_ctx().raw_data.get_random_remote_path(filename)
        protocol = get_protocol(remote_path)

        # If remote_destination was not set by the user, and the configured raw data path is also local,
        # then let's optimize by not uploading.
        hash_value = hash_method if isinstance(hash_method, str) else None
        hash_method = hash_method if isinstance(hash_method, HashMethod) else None
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
