import os
import pathlib
import random
import tempfile
import typing
from typing import AsyncIterator, Optional
from uuid import UUID

import fsspec
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from obstore.exceptions import GenericError
from obstore.fsspec import register

from flyte._initialize import get_storage
from flyte._logging import logger


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
        match protocol:
            case "s3":
                # If the protocol is s3, we can use the s3 filesystem
                from flyte.storage import S3

                return S3.auto().get_fsspec_kwargs(anonymous=anonymous)
            case "gs":
                # If the protocol is gs, we can use the gs filesystem
                from flyte.storage import GCS

                return GCS.auto().get_fsspec_kwargs(anonymous=anonymous)
            case "abfs" | "abfss":
                # If the protocol is abfs or abfss, we can use the abfs filesystem
                from flyte.storage import ABFS

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


async def get(from_path: str, to_path: Optional[str | pathlib.Path] = None, recursive: bool = False, **kwargs) -> str:
    if not to_path:
        name = pathlib.Path(from_path).name
        to_path = get_random_local_path(file_path_or_file_name=name)
        logger.debug(f"Storing file from {from_path} to {to_path}")
    file_system = get_underlying_filesystem(path=from_path)
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
            # TODO: update exception to be more specific
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
        dst = await file_system._get(from_path, to_path, recursive=recursive, **kwargs)  # pylint: disable=W0212
    else:
        dst = file_system.get(from_path, to_path, recursive=recursive, **kwargs)

    if isinstance(dst, (str, pathlib.Path)):
        return dst
    return to_path


async def put(from_path: str, to_path: Optional[str] = None, recursive: bool = False, **kwargs) -> str:
    if not to_path:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        name = pathlib.Path(from_path).name if not recursive else None  # don't pass a name for folders
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
    fs = get_underlying_filesystem(path=to_path)
    file_handle = None
    if isinstance(fs, AsyncFileSystem):
        try:
            file_handle = await fs.open_async(to_path, "wb", **kwargs)
            if isinstance(data_iterable, bytes):
                await file_handle.write(data_iterable)
            else:
                async for data in data_iterable:
                    await file_handle.write(data)
            return str(to_path)
        except NotImplementedError:
            logger.debug(f"{fs} doesn't implement 'open_async', falling back to sync")
        finally:
            if file_handle is not None:
                await file_handle.close()

    with fs.open(to_path, "wb", **kwargs) as f:
        if isinstance(data_iterable, bytes):
            f.write(data_iterable)
        else:
            # If data_iterable is async iterable, iterate over it and write each chunk to the file
            async for data in data_iterable:
                f.write(data)
    return str(to_path)


async def get_stream(path: str, chunk_size=10 * 2**20, **kwargs) -> AsyncIterator[bytes]:
    """
    Get a stream of data from a remote location.
    This is useful for downloading streaming data from a remote location.
    Example usage:
    ```python
    import flyte.storage as storage
    obj = storage.get_stream(path="s3://my_bucket/my_file.txt")
    ```

    :param path: Path to the remote location where the data will be downloaded.
    :param kwargs: Additional arguments to be passed to the underlying filesystem.
    :param chunk_size: Size of each chunk to be read from the file.
    :return: An async iterator that yields chunks of data.
    """
    fs = get_underlying_filesystem(path=path, **kwargs)
    file_size = fs.info(path)["size"]
    total_read = 0
    file_handle = None
    try:
        if isinstance(fs, AsyncFileSystem):
            file_handle = await fs.open_async(path, "rb")
            while chunk := await file_handle.read(min(chunk_size, file_size - total_read)):
                total_read += len(chunk)
                yield chunk
            return
    except NotImplementedError:
        logger.debug(f"{fs} doesn't implement 'open_async', falling back to sync")
    finally:
        if file_handle is not None:
            file_handle.close()

    # Sync fallback
    with fs.open(path, "rb") as file_handle:
        while chunk := file_handle.read(min(chunk_size, file_size - total_read)):
            total_read += len(chunk)
            yield chunk


def join(*paths: str) -> str:
    """
    Join multiple paths together. This is a wrapper around os.path.join.
    # TODO replace with proper join with fsspec root etc

    :param paths: Paths to be joined.
    """
    return str(os.path.join(*paths))


register(["s3", "gs", "abfs", "abfss"], asynchronous=True)
