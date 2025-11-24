from __future__ import annotations

import os
import pathlib
import random
import tempfile
import typing
from typing import AsyncGenerator, Optional
from uuid import UUID

import fsspec
import obstore
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from obstore.exceptions import GenericError
from obstore.fsspec import register

from flyte._initialize import get_storage
from flyte._logging import logger
from flyte.errors import InitializationError, OnlyAsyncIOSupportedError

if typing.TYPE_CHECKING:
    from obstore import AsyncReadableFile, AsyncWritableFile

_OBSTORE_SUPPORTED_PROTOCOLS = ["s3", "gs", "abfs", "abfss"]


def _is_obstore_supported_protocol(protocol: str) -> bool:
    """
    Check if the given protocol is supported by obstore.
    :param protocol: Protocol to check.
    :return: True if the protocol is supported, False otherwise.
    """
    return protocol in _OBSTORE_SUPPORTED_PROTOCOLS


def is_remote(path: typing.Union[pathlib.Path | str]) -> bool:
    """
    Let's find a replacement
    """
    protocol = get_protocol(str(path))
    if protocol is None:
        return False
    return protocol != "file"


def strip_file_header(path: str) -> str:
    """
    Drops file:// if it exists from the file
    """
    if path.startswith("file://"):
        return path.replace("file://", "", 1)
    return path


def get_random_local_path(file_path_or_file_name: pathlib.Path | str | None = None) -> pathlib.Path:
    """
    Use file_path_or_file_name, when you want a random directory, but want to preserve the leaf file name
    """
    local_tmp = pathlib.Path(tempfile.mkdtemp(prefix="flyte-tmp-"))
    key = UUID(int=random.getrandbits(128)).hex
    tmp_folder = local_tmp / key
    tail = ""
    if file_path_or_file_name:
        _, tail = os.path.split(file_path_or_file_name)
    if tail:
        tmp_folder.mkdir(parents=True, exist_ok=True)
        return tmp_folder / tail
    local_tmp.mkdir(parents=True, exist_ok=True)
    return tmp_folder


def get_random_local_directory() -> pathlib.Path:
    """
    :return: a random directory
    :rtype: pathlib.Path
    """
    _dir = get_random_local_path(None)
    pathlib.Path(_dir).mkdir(parents=True, exist_ok=True)
    return _dir


def get_configured_fsspec_kwargs(
    protocol: typing.Optional[str] = None, anonymous: bool = False
) -> typing.Dict[str, typing.Any]:
    if protocol:
        # Try to get storage config safely - may not be initialized for local operations
        try:
            storage_config = get_storage()
        except InitializationError:
            storage_config = None

        match protocol:
            case "s3":
                # If the protocol is s3, we can use the s3 filesystem
                from flyte.storage import S3

                if storage_config and isinstance(storage_config, S3):
                    return storage_config.get_fsspec_kwargs(anonymous=anonymous)

                return S3.auto().get_fsspec_kwargs(anonymous=anonymous)
            case "gs":
                # If the protocol is gs, we can use the gs filesystem
                from flyte.storage import GCS

                if storage_config and isinstance(storage_config, GCS):
                    return storage_config.get_fsspec_kwargs(anonymous=anonymous)

                return GCS.auto().get_fsspec_kwargs(anonymous=anonymous)
            case "abfs" | "abfss":
                # If the protocol is abfs or abfss, we can use the abfs filesystem
                from flyte.storage import ABFS

                if storage_config and isinstance(storage_config, ABFS):
                    return storage_config.get_fsspec_kwargs(anonymous=anonymous)

                return ABFS.auto().get_fsspec_kwargs(anonymous=anonymous)
            case _:
                return {}

    # If no protocol, return args from storage config if set
    storage_config = get_storage()
    if storage_config:
        return storage_config.get_fsspec_kwargs(anonymous)

    return {}


def get_underlying_filesystem(
    protocol: typing.Optional[str] = None,
    anonymous: bool = False,
    path: typing.Optional[str] = None,
    **kwargs,
) -> fsspec.AbstractFileSystem:
    if protocol is None:
        # If protocol is None, get it from the path
        protocol = get_protocol(path)

    configured_kwargs = get_configured_fsspec_kwargs(protocol, anonymous=anonymous)
    configured_kwargs.update(kwargs)

    return fsspec.filesystem(protocol, **configured_kwargs)


def _get_anonymous_filesystem(from_path):
    """Get the anonymous file system if needed."""
    return get_underlying_filesystem(get_protocol(from_path), anonymous=True, asynchronous=True)


async def _get_obstore_bypass(from_path: str, to_path: str | pathlib.Path, recursive: bool = False, **kwargs) -> str:
    from obstore.store import ObjectStore

    from flyte.storage._parallel_reader import ObstoreParallelReader

    fs = get_underlying_filesystem(path=from_path)
    bucket, prefix = fs._split_path(from_path)  # pylint: disable=W0212
    store: ObjectStore = fs._construct_store(bucket)

    download_kwargs = {}
    if "chunk_size" in kwargs:
        download_kwargs["chunk_size"] = kwargs["chunk_size"]
    if "max_concurrency" in kwargs:
        download_kwargs["max_concurrency"] = kwargs["max_concurrency"]

    reader = ObstoreParallelReader(store, **download_kwargs)
    target_path = pathlib.Path(to_path) if isinstance(to_path, str) else to_path

    # if recursive, just download the prefix to the target path
    if recursive:
        logger.debug(f"Downloading recursively {prefix=} to {target_path=}")
        await reader.download_files(
            prefix,
            target_path,
        )
        return str(to_path)

    # if not recursive, we need to split out the file name from the prefix
    else:
        path_for_reader = pathlib.Path(prefix).name
        final_prefix = pathlib.Path(prefix).parent
        logger.debug(f"Downloading single file {final_prefix=}, {path_for_reader=} to {target_path=}")
        await reader.download_files(
            final_prefix,
            target_path.parent,
            path_for_reader,
            destination_file_name=target_path.name,
        )
        return str(target_path)


async def get(from_path: str, to_path: Optional[str | pathlib.Path] = None, recursive: bool = False, **kwargs) -> str:
    if not to_path:
        name = pathlib.Path(from_path).name  # may need to be adjusted for windows
        to_path = get_random_local_path(file_path_or_file_name=name)
        logger.debug(f"Storing file from {from_path} to {to_path}")
    else:
        # Only apply directory logic for single files (not recursive)
        if not recursive:
            to_path_str = str(to_path)
            # Check for trailing separator BEFORE converting to Path (which normalizes and removes it)
            ends_with_sep = to_path_str.endswith(os.sep)
            to_path_obj = pathlib.Path(to_path)

            # If path ends with os.sep or is an existing directory, append source filename
            if ends_with_sep or (to_path_obj.exists() and to_path_obj.is_dir()):
                source_filename = pathlib.Path(from_path).name  # may need to be adjusted for windows
                to_path = to_path_obj / source_filename
        # For recursive=True, keep to_path as-is (it's the destination directory for contents)

    file_system = get_underlying_filesystem(path=from_path)

    # Check if we should use obstore bypass
    if (
        _is_obstore_supported_protocol(file_system.protocol)
        and hasattr(file_system, "_split_path")
        and hasattr(file_system, "_construct_store")
        and recursive
    ):
        return await _get_obstore_bypass(from_path, to_path, recursive, **kwargs)

    try:
        return await _get_from_filesystem(file_system, from_path, to_path, recursive=recursive, **kwargs)
    except (OSError, GenericError) as oe:
        logger.debug(f"Error in getting {from_path} to {to_path} rec {recursive} {oe}")
        if isinstance(file_system, AsyncFileSystem):
            try:
                exists = await file_system._exists(from_path)  # pylint: disable=W0212
            except GenericError:
                # for obstore, as it does not raise FileNotFoundError in fsspec but GenericError
                # force it to try get_filesystem(anonymous=True)
                exists = True
        else:
            exists = file_system.exists(from_path)
        if not exists:
            raise AssertionError(f"Unable to load data from {from_path}")
        file_system = _get_anonymous_filesystem(from_path)
        logger.debug(f"Attempting anonymous get with {file_system}")
        return await _get_from_filesystem(file_system, from_path, to_path, recursive=recursive, **kwargs)


async def _get_from_filesystem(
    file_system: fsspec.AbstractFileSystem,
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path,
    recursive: bool,
    **kwargs,
):
    if isinstance(file_system, AsyncFileSystem):
        dst = await file_system._get(str(from_path), str(to_path), recursive=recursive, **kwargs)  # pylint: disable=W0212
    else:
        dst = file_system.get(str(from_path), str(to_path), recursive=recursive, **kwargs)

    if isinstance(dst, (str, pathlib.Path)):
        return dst
    return str(to_path)


async def put(from_path: str, to_path: Optional[str] = None, recursive: bool = False, **kwargs) -> str:
    if not to_path:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        name = pathlib.Path(from_path).name
        to_path = ctx.raw_data.get_random_remote_path(file_name=name)

    file_system = get_underlying_filesystem(path=to_path)
    from_path = strip_file_header(from_path)
    if isinstance(file_system, AsyncFileSystem):
        dst = await file_system._put(from_path, to_path, recursive=recursive, **kwargs)  # pylint: disable=W0212
    else:
        dst = file_system.put(from_path, to_path, recursive=recursive, **kwargs)
    if isinstance(dst, (str, pathlib.Path)):
        return str(dst)
    else:
        return to_path


async def _open_obstore_bypass(path: str, mode: str = "rb", **kwargs) -> AsyncReadableFile | AsyncWritableFile:
    """
    Simple obstore bypass for opening files. No fallbacks, obstore only.
    """
    from obstore.store import ObjectStore

    fs = get_underlying_filesystem(path=path)
    bucket, file_path = fs._split_path(path)  # pylint: disable=W0212
    store: ObjectStore = fs._construct_store(bucket)

    file_handle: AsyncReadableFile | AsyncWritableFile

    if "w" in mode:
        attributes = kwargs.pop("attributes", {})
        file_handle = obstore.open_writer_async(store, file_path, attributes=attributes)
    else:  # read mode
        buffer_size = kwargs.pop("buffer_size", 10 * 2**20)
        file_handle = await obstore.open_reader_async(store, file_path, buffer_size=buffer_size)

    return file_handle


async def open(path: str, mode: str = "rb", **kwargs) -> AsyncReadableFile | AsyncWritableFile:
    """
    Asynchronously open a file and return an async context manager.
    This function checks if the underlying filesystem supports obstore bypass.
    If it does, it uses obstore to open the file. Otherwise, it falls back to
    the standard _open function which uses AsyncFileSystem.

    It will raise NotImplementedError if neither obstore nor AsyncFileSystem is supported.
    """
    fs = get_underlying_filesystem(path=path)

    # Check if we should use obstore bypass
    if _is_obstore_supported_protocol(fs.protocol) and hasattr(fs, "_split_path") and hasattr(fs, "_construct_store"):
        return await _open_obstore_bypass(path, mode, **kwargs)

    # Fallback to normal open
    if isinstance(fs, AsyncFileSystem):
        return await fs.open_async(path, mode, **kwargs)

    raise OnlyAsyncIOSupportedError(f"Filesystem {fs} does not support async operations")


async def put_stream(
    data_iterable: typing.AsyncIterable[bytes] | bytes, *, name: str | None = None, to_path: str | None = None, **kwargs
) -> str:
    """
    Put a stream of data to a remote location. This is useful for streaming data to a remote location.
    Example usage:
    ```python
    import flyte.storage as storage
    storage.put_stream(iter([b'hello']), name="my_file.txt")
    OR
    storage.put_stream(iter([b'hello']), to_path="s3://my_bucket/my_file.txt")
    ```

    :param data_iterable: Iterable of bytes to be streamed.
    :param name: Name of the file to be created. If not provided, a random name will be generated.
    :param to_path: Path to the remote location where the data will be stored.
    :param kwargs: Additional arguments to be passed to the underlying filesystem.
    :rtype: str
    :return: The path to the remote location where the data was stored.
    """
    if not to_path:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        to_path = ctx.raw_data.get_random_remote_path(file_name=name)

    # Check if we should use obstore bypass
    fs = get_underlying_filesystem(path=to_path)
    try:
        file_handle = typing.cast("AsyncWritableFile", await open(to_path, "wb", **kwargs))
        if isinstance(data_iterable, bytes):
            await file_handle.write(data_iterable)
        else:
            async for data in data_iterable:
                await file_handle.write(data)
        await file_handle.close()
        return str(to_path)
    except OnlyAsyncIOSupportedError:
        pass

    # Fallback to normal open
    file_handle_io: typing.IO = fs.open(to_path, mode="wb", **kwargs)
    if isinstance(data_iterable, bytes):
        file_handle_io.write(data_iterable)
    else:
        async for data in data_iterable:
            file_handle_io.write(data)
    file_handle_io.close()

    return str(to_path)


async def get_stream(path: str, chunk_size=10 * 2**20, **kwargs) -> AsyncGenerator[bytes, None]:
    """
    Get a stream of data from a remote location.
    This is useful for downloading streaming data from a remote location.
    Example usage:
    ```python
    import flyte.storage as storage
    async for chunk in storage.get_stream(path="s3://my_bucket/my_file.txt"):
        process(chunk)
    ```

    :param path: Path to the remote location where the data will be downloaded.
    :param kwargs: Additional arguments to be passed to the underlying filesystem.
    :param chunk_size: Size of each chunk to be read from the file.
    :return: An async iterator that yields chunks of bytes.
    """
    # Check if we should use obstore bypass
    fs = get_underlying_filesystem(path=path)
    if _is_obstore_supported_protocol(fs.protocol) and hasattr(fs, "_split_path") and hasattr(fs, "_construct_store"):
        # Set buffer_size for obstore if chunk_size is provided
        if "buffer_size" not in kwargs:
            kwargs["buffer_size"] = chunk_size
        file_handle = typing.cast("AsyncReadableFile", await _open_obstore_bypass(path, "rb", **kwargs))
        while chunk := await file_handle.read():
            yield bytes(chunk)
        return

    # Fallback to normal open
    if "block_size" not in kwargs:
        kwargs["block_size"] = chunk_size

    if isinstance(fs, AsyncFileSystem):
        file_handle = await fs.open_async(path, "rb", **kwargs)
        while chunk := await file_handle.read():
            yield chunk
        await file_handle.close()
        return

    file_handle = fs.open(path, "rb", **kwargs)
    while chunk := file_handle.read():
        yield chunk
    file_handle.close()


def join(*paths: str) -> str:
    """
    Join multiple paths together. This is a wrapper around os.path.join.
    # TODO replace with proper join with fsspec root etc

    :param paths: Paths to be joined.
    """
    return str(os.path.join(*paths))


async def exists(path: str, **kwargs) -> bool:
    """
    Check if a path exists.

    :param path: Path to be checked.
    :param kwargs: Additional arguments to be passed to the underlying filesystem.
    :return: True if the path exists, False otherwise.
    """
    try:
        fs = get_underlying_filesystem(path=path, **kwargs)
        if isinstance(fs, AsyncFileSystem):
            _ = await fs._info(path)
            return True
        _ = fs.info(path)
        return True
    except FileNotFoundError:
        return False


def exists_sync(path: str, **kwargs) -> bool:
    try:
        fs = get_underlying_filesystem(path=path, **kwargs)
        _ = fs.info(path)
        return True
    except FileNotFoundError:
        return False


register(_OBSTORE_SUPPORTED_PROTOCOLS, asynchronous=True)
