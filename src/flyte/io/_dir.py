from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

from flyteidl2.core import literals_pb2, types_pb2
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from mashumaro.types import SerializableType
from pydantic import BaseModel, model_validator

import flyte.storage as storage
from flyte._context import internal_ctx
from flyte.io._file import File
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError

# Type variable for the directory format
T = TypeVar("T")


class Dir(BaseModel, Generic[T], SerializableType):
    """
    A generic directory class representing a directory with files of a specified format.
    Provides both async and sync interfaces for directory operations. All methods without _sync suffix are async.

    The class should be instantiated using one of the class methods. The constructor should only be used to
    instantiate references to existing remote directories.

    The generic type T represents the format of the files in the directory.

    Important methods:
    - `from_existing_remote`: Create a Dir object referencing an existing remote directory.
    - `from_local` / `from_local_sync`: Upload a local directory to remote storage.

    **Asynchronous methods**:
    - `walk`: Asynchronously iterate through files in the directory.
    - `list_files`: Asynchronously get a list of all files (non-recursive).
    - `download`: Asynchronously download the entire directory to a local path.
    - `exists`: Asynchronously check if the directory exists.
    - `get_file`: Asynchronously get a specific file from the directory by name.

    **Synchronous methods** (suffixed with `_sync`):
    - `walk_sync`: Synchronously iterate through files in the directory.
    - `list_files_sync`: Synchronously get a list of all files (non-recursive).
    - `download_sync`: Synchronously download the entire directory to a local path.
    - `exists_sync`: Synchronously check if the directory exists.
    - `get_file_sync`: Synchronously get a specific file from the directory by name.

    Example: Walk through directory files recursively (Async).

    ```python
    @env.task
    async def process_all_files(d: Dir) -> int:
        file_count = 0
        async for file in d.walk(recursive=True):
            async with file.open("rb") as f:
                content = await f.read()
                # Process content
                file_count += 1
        return file_count
    ```

    Example: Walk through directory files recursively (Sync).

    ```python
    @env.task
    def process_all_files_sync(d: Dir) -> int:
        file_count = 0
        for file in d.walk_sync(recursive=True):
            with file.open_sync("rb") as f:
                content = f.read()
                # Process content
                file_count += 1
        return file_count
    ```

    Example: List files in directory (Async).

    ```python
    @env.task
    async def count_files(d: Dir) -> int:
        files = await d.list_files()
        return len(files)
    ```

    Example: List files in directory (Sync).

    ```python
    @env.task
    def count_files_sync(d: Dir) -> int:
        files = d.list_files_sync()
        return len(files)
    ```

    Example: Get a specific file from directory (Async).

    ```python
    @env.task
    async def read_config_file(d: Dir) -> str:
        config_file = await d.get_file("config.json")
        if config_file:
            async with config_file.open("rb") as f:
                return (await f.read()).decode("utf-8")
        return "Config not found"
    ```

    Example: Get a specific file from directory (Sync).

    ```python
    @env.task
    def read_config_file_sync(d: Dir) -> str:
        config_file = d.get_file_sync("config.json")
        if config_file:
            with config_file.open_sync("rb") as f:
                return f.read().decode("utf-8")
        return "Config not found"
    ```

    Example: Upload a local directory to remote storage (Async).

    ```python
    @env.task
    async def upload_directory() -> Dir:
        # Create local directory with files
        os.makedirs("/tmp/my_data", exist_ok=True)
        with open("/tmp/my_data/file1.txt", "w") as f:
            f.write("data1")
        # Upload to remote storage
        return await Dir.from_local("/tmp/my_data/")
    ```

    Example: Upload a local directory to remote storage (Sync).

    ```python
    @env.task
    def upload_directory_sync() -> Dir:
        # Create local directory with files
        os.makedirs("/tmp/my_data", exist_ok=True)
        with open("/tmp/my_data/file1.txt", "w") as f:
            f.write("data1")
        # Upload to remote storage
        return Dir.from_local_sync("/tmp/my_data/")
    ```

    Example: Download a directory to local storage (Async).

    ```python
    @env.task
    async def download_directory(d: Dir) -> str:
        local_path = await d.download()
        # Process files in local directory
        return local_path
    ```

    Example: Download a directory to local storage (Sync).

    ```python
    @env.task
    def download_directory_sync(d: Dir) -> str:
        local_path = d.download_sync()
        # Process files in local directory
        return local_path
    ```

    Example: Reference an existing remote directory.

    ```python
    @env.task
    async def process_existing_dir() -> int:
        d = Dir.from_existing_remote("s3://my-bucket/data/")
        files = await d.list_files()
        return len(files)
    ```

    Example: Check if directory exists (Async).

    ```python
    @env.task
    async def check_directory(d: Dir) -> bool:
        return await d.exists()
    ```

    Example: Check if directory exists (Sync).

    ```python
    @env.task
    def check_directory_sync(d: Dir) -> bool:
        return d.exists_sync()
    ```

    Args:
        path: The path to the directory (can be local or remote)
        name: Optional name for the directory (defaults to basename of path)
    """

    # Represents either a local or remote path.
    path: str
    name: Optional[str] = None
    format: str = ""
    hash: Optional[str] = None

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
        """Internal: Serialize Dir to dictionary. Not intended for direct use."""
        pyd_dump = self.model_dump()
        return pyd_dump

    @classmethod
    def _deserialize(cls, file_dump: Dict[str, Optional[str]]) -> Dir:
        """Internal: Deserialize Dir from dictionary. Not intended for direct use."""
        return cls.model_validate(file_dump)

    @classmethod
    def schema_match(cls, incoming: dict):
        """Internal: Check if incoming schema matches Dir schema. Not intended for direct use."""
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

        Use this to iterate through all files in a directory. Each yielded File can be read directly without
        downloading.

        Example (Async - Recursive):

        ```python
        @env.task
        async def list_all_files(d: Dir) -> list[str]:
            file_names = []
            async for file in d.walk(recursive=True):
                file_names.append(file.name)
            return file_names
        ```

        Example (Async - Non-recursive):

        ```python
        @env.task
        async def list_top_level_files(d: Dir) -> list[str]:
            file_names = []
            async for file in d.walk(recursive=False):
                file_names.append(file.name)
            return file_names
        ```

        Example (Async - With max depth):

        ```python
        @env.task
        async def list_files_max_depth(d: Dir) -> list[str]:
            file_names = []
            async for file in d.walk(recursive=True, max_depth=2):
                file_names.append(file.name)
            return file_names
        ```

        Args:
            recursive: If True, recursively walk subdirectories. If False, only list files in the top-level directory.
            max_depth: Maximum depth for recursive walking. If None, walk through all subdirectories.

        Yields:
            File objects for each file found in the directory
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

        Use this in non-async tasks to iterate through all files in a directory.

        Example (Sync - Recursive):

        ```python
        @env.task
        def list_all_files_sync(d: Dir) -> list[str]:
            file_names = []
            for file in d.walk_sync(recursive=True):
                file_names.append(file.name)
            return file_names
        ```

        Example (Sync - With file pattern):

        ```python
        @env.task
        def list_text_files(d: Dir) -> list[str]:
            file_names = []
            for file in d.walk_sync(recursive=True, file_pattern="*.txt"):
                file_names.append(file.name)
            return file_names
        ```

        Example (Sync - Non-recursive with max depth):

        ```python
        @env.task
        def list_files_limited(d: Dir) -> list[str]:
            file_names = []
            for file in d.walk_sync(recursive=True, max_depth=2):
                file_names.append(file.name)
            return file_names
        ```

        Args:
            recursive: If True, recursively walk subdirectories. If False, only list files in the top-level directory.
            file_pattern: Glob pattern to filter files (e.g., "*.txt", "*.csv"). Default is "*" (all files).
            max_depth: Maximum depth for recursive walking. If None, walk through all subdirectories.

        Yields:
            File objects for each file found in the directory
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

        Use this when you need a list of all files in the top-level directory at once.

        Returns:
            A list of File objects for files in the top-level directory

        Example (Async):

        ```python
        @env.task
        async def count_files(d: Dir) -> int:
            files = await d.list_files()
            return len(files)
        ```

        Example (Async - Process files):

        ```python
        @env.task
        async def process_all_files(d: Dir) -> list[str]:
            files = await d.list_files()
            contents = []
            for file in files:
                async with file.open("rb") as f:
                    content = await f.read()
                    contents.append(content.decode("utf-8"))
            return contents
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

        Use this in non-async tasks when you need a list of all files in the top-level directory at once.

        Returns:
            A list of File objects for files in the top-level directory

        Example (Sync):

        ```python
        @env.task
        def count_files_sync(d: Dir) -> int:
            files = d.list_files_sync()
            return len(files)
        ```

        Example (Sync - Process files):

        ```python
        @env.task
        def process_all_files_sync(d: Dir) -> list[str]:
            files = d.list_files_sync()
            contents = []
            for file in files:
                with file.open_sync("rb") as f:
                    content = f.read()
                    contents.append(content.decode("utf-8"))
            return contents
        ```
        """
        return list(self.walk_sync(recursive=False))

    async def download(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Asynchronously download the entire directory to a local path.

        Use this when you need to download all files in a directory to your local filesystem for processing.

        Example (Async):

        ```python
        @env.task
        async def download_directory(d: Dir) -> str:
            local_dir = await d.download()
            # Process files in the local directory
            return local_dir
        ```

        Example (Async - Download to specific path):

        ```python
        @env.task
        async def download_to_path(d: Dir) -> str:
            local_dir = await d.download("/tmp/my_data/")
            return local_dir
        ```

        Args:
            local_path: The local path to download the directory to. If None, a temporary
                       directory will be used and a path will be generated.

        Returns:
            The absolute path to the downloaded directory
        """
        # If no local_path specified, create a unique path + append source directory name
        if local_path is None:
            unique_path = storage.get_random_local_path()
            source_dirname = Path(self.path).name  # will need to be updated for windows
            local_dest = str(Path(unique_path) / source_dirname)
        else:
            # If local_path is specified, use it directly (contents go into it)
            local_dest = str(local_path)

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
                return local_dest
        return await storage.get(self.path, local_dest, recursive=True)

    def download_sync(self, local_path: Optional[Union[str, Path]] = None) -> str:
        """
        Synchronously download the entire directory to a local path.

        Use this in non-async tasks when you need to download all files in a directory to your local filesystem.

        Example (Sync):

        ```python
        @env.task
        def download_directory_sync(d: Dir) -> str:
            local_dir = d.download_sync()
            # Process files in the local directory
            return local_dir
        ```

        Example (Sync - Download to specific path):

        ```python
        @env.task
        def download_to_path_sync(d: Dir) -> str:
            local_dir = d.download_sync("/tmp/my_data/")
            return local_dir
        ```
        Args:
            local_path: The local path to download the directory to. If None, a temporary
                       directory will be used and a path will be generated.

        Returns:
            The absolute path to the downloaded directory
        """
        # If no local_path specified, create a unique path + append source directory name
        if local_path is None:
            unique_path = storage.get_random_local_path()
            source_dirname = Path(self.path).name
            local_dest = str(Path(unique_path) / source_dirname)
        else:
            # If local_path is specified, use it directly (contents go into it)
            local_dest = str(local_path)

        if not storage.is_remote(self.path):
            if not local_path or local_path == self.path:
                # Skip copying
                return self.path
            else:
                # Shell out to a thread to copy
                import shutil

                shutil.copytree(self.path, local_dest, dirs_exist_ok=True)
                return local_dest

        fs = storage.get_underlying_filesystem(path=self.path)
        fs.get(self.path, local_dest, recursive=True)
        return local_dest

    @classmethod
    async def from_local(
        cls,
        local_path: Union[str, Path],
        remote_destination: Optional[str] = None,
        dir_cache_key: Optional[str] = None,
    ) -> Dir[T]:
        """
        Asynchronously create a new Dir by uploading a local directory to remote storage.

        Use this in async tasks when you have a local directory that needs to be uploaded to remote storage.

        Example (Async):

        ```python
        @env.task
        async def upload_local_directory() -> Dir:
            # Create a local directory with files
            os.makedirs("/tmp/data_dir", exist_ok=True)
            with open("/tmp/data_dir/file1.txt", "w") as f:
                f.write("data1")

            # Upload to remote storage
            remote_dir = await Dir.from_local("/tmp/data_dir/")
            return remote_dir
        ```

        Example (Async - With specific destination):

        ```python
        @env.task
        async def upload_to_specific_path() -> Dir:
            remote_dir = await Dir.from_local("/tmp/data_dir/", "s3://my-bucket/data/")
            return remote_dir
        ```

        Example (Async - With cache key):

        ```python
        @env.task
        async def upload_with_cache_key() -> Dir:
            remote_dir = await Dir.from_local("/tmp/data_dir/", dir_cache_key="my_cache_key_123")
            return remote_dir
        ```
        Args:
            local_path: Path to the local directory
            remote_destination: Optional remote path to store the directory. If None, a path will be automatically
              generated.
            dir_cache_key: Optional precomputed hash value to use for cache key computation when this Dir is used
                          as an input to discoverable tasks. If not specified, the cache key will be based on
                          directory attributes.

        Returns:
            A new Dir instance pointing to the uploaded directory
        """
        local_path_str = str(local_path)
        dirname = os.path.basename(os.path.normpath(local_path_str))
        resolved_remote_path = remote_destination or internal_ctx().raw_data.get_random_remote_path(dirname)
        protocol = get_protocol(resolved_remote_path)

        # Shortcut for local, don't copy and just return
        if "file" in protocol and remote_destination is None:
            output_path = str(Path(local_path).absolute())
            return cls(path=output_path, name=dirname, hash=dir_cache_key)

        # todo: in the future, mirror File and set the file to_path here
        output_path = await storage.put(from_path=local_path_str, to_path=remote_destination, recursive=True)
        return cls(path=output_path, name=dirname, hash=dir_cache_key)

    @classmethod
    def from_local_sync(
        cls,
        local_path: Union[str, Path],
        remote_destination: Optional[str] = None,
        dir_cache_key: Optional[str] = None,
    ) -> Dir[T]:
        """
        Synchronously create a new Dir by uploading a local directory to remote storage.

        Use this in non-async tasks when you have a local directory that needs to be uploaded to remote storage.

        Example (Sync):

        ```python
        @env.task
        def upload_local_directory_sync() -> Dir:
            # Create a local directory with files
            os.makedirs("/tmp/data_dir", exist_ok=True)
            with open("/tmp/data_dir/file1.txt", "w") as f:
                f.write("data1")

            # Upload to remote storage
            remote_dir = Dir.from_local_sync("/tmp/data_dir/")
            return remote_dir
        ```

        Example (Sync - With specific destination):

        ```python
        @env.task
        def upload_to_specific_path_sync() -> Dir:
            remote_dir = Dir.from_local_sync("/tmp/data_dir/", "s3://my-bucket/data/")
            return remote_dir
        ```

        Example (Sync - With cache key):

        ```python
        @env.task
        def upload_with_cache_key_sync() -> Dir:
            remote_dir = Dir.from_local_sync("/tmp/data_dir/", dir_cache_key="my_cache_key_123")
            return remote_dir
        ```

        Args:
            local_path: Path to the local directory
            remote_destination: Optional remote path to store the directory. If None, a path will be automatically
              generated.
            dir_cache_key: Optional precomputed hash value to use for cache key computation when this Dir is used
                          as an input to discoverable tasks. If not specified, the cache key will be based on
                          directory attributes.

        Returns:
            A new Dir instance pointing to the uploaded directory
        """
        local_path_str = str(local_path)
        dirname = os.path.basename(os.path.normpath(local_path_str))

        resolved_remote_path = remote_destination or internal_ctx().raw_data.get_random_remote_path(dirname)
        protocol = get_protocol(resolved_remote_path)

        # Shortcut for local, don't copy and just return
        if "file" in protocol and remote_destination is None:
            output_path = str(Path(local_path).absolute())
            return cls(path=output_path, name=dirname, hash=dir_cache_key)

        fs = storage.get_underlying_filesystem(path=resolved_remote_path)
        fs.put(local_path_str, resolved_remote_path, recursive=True)
        return cls(path=resolved_remote_path, name=dirname, hash=dir_cache_key)

    @classmethod
    def from_existing_remote(cls, remote_path: str, dir_cache_key: Optional[str] = None) -> Dir[T]:
        """
        Create a Dir reference from an existing remote directory.

        Use this when you want to reference a directory that already exists in remote storage without uploading it.

        Example:

        ```python
        @env.task
        async def process_existing_directory() -> int:
            d = Dir.from_existing_remote("s3://my-bucket/data/")
            files = await d.list_files()
            return len(files)
        ```

        Example (With cache key):

        ```python
        @env.task
        async def process_with_cache_key() -> int:
            d = Dir.from_existing_remote("s3://my-bucket/data/", dir_cache_key="abc123")
            files = await d.list_files()
            return len(files)
        ```

        Args:
            remote_path: The remote path to the existing directory
            dir_cache_key: Optional hash value to use for cache key computation. If not specified,
                          the cache key will be computed based on the directory's attributes.

        Returns:
            A new Dir instance pointing to the existing remote directory
        """
        return cls(path=remote_path, hash=dir_cache_key)

    async def exists(self) -> bool:
        """
        Asynchronously check if the directory exists.

        Returns:
            True if the directory exists, False otherwise

        Example (Async):

        ```python
        @env.task
        async def check_directory(d: Dir) -> bool:
            if await d.exists():
                print("Directory exists!")
                return True
            return False
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

        Use this in non-async tasks or when you need synchronous directory existence checking.

        Returns:
            True if the directory exists, False otherwise

        Example (Sync):

        ```python
        @env.task
        def check_directory_sync(d: Dir) -> bool:
            if d.exists_sync():
                print("Directory exists!")
                return True
            return False
        ```
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        return fs.exists(self.path)

    async def get_file(self, file_name: str) -> Optional[File[T]]:
        """
        Asynchronously get a specific file from the directory by name.

        Use this when you know the name of a specific file in the directory you want to access.

        Example (Async):

        ```python
        @env.task
        async def read_specific_file(d: Dir) -> str:
            file = await d.get_file("data.csv")
            if file:
                async with file.open("rb") as f:
                    content = await f.read()
                    return content.decode("utf-8")
            return "File not found"
        ```

        Args:
            file_name: The name of the file to get

        Returns:
            A File instance if the file exists, None otherwise
        """
        fs = storage.get_underlying_filesystem(path=self.path)
        file_path = fs.sep.join([self.path, file_name])
        file = File[T](path=file_path)

        if fs.exists(file_path):
            return file
        return None

    def get_file_sync(self, file_name: str) -> Optional[File[T]]:
        """
        Synchronously get a specific file from the directory by name.

        Use this in non-async tasks when you know the name of a specific file in the directory you want to access.

        Example (Sync):

        ```python
        @env.task
        def read_specific_file_sync(d: Dir) -> str:
            file = d.get_file_sync("data.csv")
            if file:
                with file.open_sync("rb") as f:
                    content = f.read()
                    return content.decode("utf-8")
            return "File not found"
        ```

        Args:
            file_name: The name of the file to get

        Returns:
            A File instance if the file exists, None otherwise
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
            ),
            hash=python_val.hash if python_val.hash else None,
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
        hash_value = lv.hash if lv.hash else None
        f: Dir = Dir(path=uri, name=filename, format=lv.scalar.blob.metadata.type.format, hash=hash_value)
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
