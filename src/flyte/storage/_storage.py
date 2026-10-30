from __future__ import annotations

import os
import pathlib
import random
import tempfile
import typing
from typing import AsyncGenerator, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import fsspec
import obstore
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from obstore.exceptions import GenericError
from obstore.fsspec import register
from obstore.store import ObjectStore

from flyte._initialize import get_storage
from flyte._logging import logger
from flyte.errors import InitializationError, OnlyAsyncIOSupportedError

from ._remote_fs import FlyteFS

if typing.TYPE_CHECKING:
    from obstore import AsyncReadableFile, AsyncWritableFile

_OBSTORE_SUPPORTED_PROTOCOLS = ["s3", "gs", "abfs", "abfss"]
BATCH_SIZE = int(os.getenv("FLYTE_IO_BATCH_SIZE", str(32)))

# Multipart upload sizing. GCS and S3 both hard-cap a multipart upload at 10,000 parts, so a fixed
# part size imposes a hard file-size ceiling (obstore's 5 MiB default => ~48.8 GiB). We instead
# auto-scale the part size with the file size so uploads never run out of part numbers, while small
# files keep small parts.
#
# The defaults mirror obstore's own put/put_async defaults (chunk_size=5242880, max_concurrency=12),
# defined here:
# https://github.com/developmentseed/obstore/blob/c6279ce19f73eef89321179d55930d0dd7ba6a62/obstore/python/obstore/store.py#L500-L501
#
# Both can be overridden with environment variables, or per call with the `chunk_size` /
# `max_concurrency` arguments of put() and put_stream(). Each upload request is one part, so a larger
# part size means fewer requests for the same bytes (useful when the object store throttles on request
# rate). Each part in flight is held in memory, so peak upload memory per pod is roughly
# part size x max concurrency x FLYTE_IO_BATCH_SIZE.
_S3_MIN_PART_SIZE = 5 * 2**20  # S3 rejects non-final multipart parts smaller than 5 MiB


def _int_from_env(name: str, default: int, minimum: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    parsed = int(value)
    if parsed < minimum:
        logger.warning(f"{name}={parsed} is below the minimum of {minimum}, using {minimum}")
        return minimum
    return parsed


# Floor for the part size; large files scale above it.
_UPLOAD_CHUNK_FLOOR = _int_from_env("FLYTE_IO_UPLOAD_CHUNK_SIZE", _S3_MIN_PART_SIZE, _S3_MIN_PART_SIZE)
# Number of parts of a single file uploaded at once.
_UPLOAD_MAX_CONCURRENCY = _int_from_env("FLYTE_IO_UPLOAD_MAX_CONCURRENCY", 12, 1)
_MAX_SAFE_PARTS = 9000  # margin below the cloud 10,000-part hard limit
_STREAM_WRITE_BUFFER_SIZE = 10 * 2**20  # obstore open_writer default; each multipart part is this size


def _resolve_chunk_floor(chunk_size: int | None) -> int:
    """Floor for the part size: the per-call `chunk_size` if given, else the env-configured default."""
    if chunk_size is None:
        return _UPLOAD_CHUNK_FLOOR
    if chunk_size < _S3_MIN_PART_SIZE:
        logger.warning(
            f"chunk_size={chunk_size} is below the minimum of {_S3_MIN_PART_SIZE}, using {_S3_MIN_PART_SIZE}"
        )
        return _S3_MIN_PART_SIZE
    return chunk_size


def _resolve_max_concurrency(max_concurrency: int | None) -> int:
    """Parts in flight per file: the per-call `max_concurrency` if given, else the env-configured default."""
    if max_concurrency is None:
        return _UPLOAD_MAX_CONCURRENCY
    return max(1, max_concurrency)


def _compute_upload_chunk_size(file_size: int | None, floor: int | None = None) -> int:
    """
    Pick a multipart part size that keeps the part count under the cloud 10,000-part limit.

    Returns the floor (`floor` if given, else 5 MiB unless FLYTE_IO_UPLOAD_CHUNK_SIZE is set) for
    small/unknown files, and scales up just enough (ceil division) for large files so
    `file_size / chunk_size <= _MAX_SAFE_PARTS`.
    """
    resolved_floor = _resolve_chunk_floor(floor)
    if not file_size:
        return resolved_floor
    needed = -(-file_size // _MAX_SAFE_PARTS)  # integer ceil division
    return max(resolved_floor, needed)


def _is_obstore_supported_protocol(protocol: str | tuple[str, ...]) -> bool:
    """
    Check if the given protocol is supported by obstore.

    Args:
        protocol: Protocol to check.

    Returns:
        True if the protocol is supported, False otherwise.
    """
    return protocol in _OBSTORE_SUPPORTED_PROTOCOLS


def _supports_obstore_bypass(file_system: typing.Any) -> bool:
    """Whether uploads to this filesystem go through the obstore native API (and so honor part sizing)."""
    return (
        _is_obstore_supported_protocol(getattr(file_system, "protocol", ""))
        and hasattr(file_system, "_split_path")
        and hasattr(file_system, "_construct_store")
    )


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
    Returns:
        pathlib.Path: a random directory
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
        protocol = get_protocol(typing.cast(str, path))

    configured_kwargs = get_configured_fsspec_kwargs(protocol, anonymous=anonymous)
    configured_kwargs.update(kwargs)

    return fsspec.filesystem(protocol, **configured_kwargs)


def _get_anonymous_filesystem(from_path):
    """Get the anonymous file system if needed."""
    return get_underlying_filesystem(get_protocol(from_path), anonymous=True, asynchronous=True)


async def _get_obstore_bypass(
    from_path: str,
    to_path: str | pathlib.Path,
    recursive: bool = False,
    exclude: list[str] | None = None,
    **kwargs,
) -> str:
    from flyte.storage._parallel_reader import ObstoreParallelReader

    # The obstore-backed filesystem is duck-typed (callers guard on hasattr for _split_path/_construct_store).
    fs = typing.cast(typing.Any, get_underlying_filesystem(path=from_path))
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
            exclude=exclude,
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
    ):
        return await _get_obstore_bypass(from_path, to_path, recursive, **kwargs)

    try:
        return await _get_from_filesystem(file_system, from_path, to_path, recursive=recursive, **kwargs)
    except (OSError, GenericError) as oe:
        logger.debug(f"Error in getting {from_path} to {to_path}, recursive: {recursive}, error: {oe}")
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


async def _put_single_obstore(
    store: ObjectStore,
    local_path: str | pathlib.Path,
    remote_key: str,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
) -> None:
    """Upload a single local file to `remote_key` with an auto-sized multipart part size."""
    # Passing a pathlib.Path streams the file from disk rather than reading it all into memory.
    await store.put_async(
        remote_key,
        pathlib.Path(local_path),
        chunk_size=_compute_upload_chunk_size(os.path.getsize(local_path), floor=chunk_size),
        max_concurrency=_resolve_max_concurrency(max_concurrency),
    )


async def _put_obstore_bypass(
    from_path: str,
    to_path: str,
    recursive: bool = False,
    batch_size: int = BATCH_SIZE,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
) -> str:
    """
    Upload via the obstore native API so we can control multipart part sizing.

    fsspec's obstore backend (`_put_file`) calls `store.put_async` without a `chunk_size`,
    leaving it pinned to obstore's 5 MiB default and thus a ~48.8 GiB hard ceiling (10,000 parts).
    This bypass computes the part size from each file's size so uploads of any size succeed. Mirrors
    the download-side `_get_obstore_bypass` / `ObstoreParallelReader` pattern.
    """
    import asyncio

    # The obstore-backed filesystem is duck-typed (callers guard on hasattr for _split_path/_construct_store).
    fs = typing.cast(typing.Any, get_underlying_filesystem(path=to_path))
    bucket, prefix = fs._split_path(to_path)  # pylint: disable=W0212
    store: ObjectStore = fs._construct_store(bucket)
    if chunk_size is not None:
        # Resolve once so an out-of-range value warns once, not once per file.
        chunk_size = _resolve_chunk_floor(chunk_size)

    if not recursive:
        logger.debug(f"Uploading single file {from_path} to {to_path}")
        await _put_single_obstore(store, from_path, prefix, chunk_size, max_concurrency)
        return to_path

    # Recursive: walk the local tree and upload each file, preserving its path relative to from_path.
    source_root = pathlib.Path(from_path)
    files = [p for p in source_root.rglob("*") if p.is_file()]
    logger.debug(f"Uploading {len(files)} files recursively from {from_path} to {to_path}")

    prefix_path = pathlib.PurePosixPath(prefix)
    semaphore = asyncio.Semaphore(batch_size)

    async def _upload(local_file: pathlib.Path) -> None:
        relative = local_file.relative_to(source_root).as_posix()
        remote_key = str(prefix_path / relative)
        async with semaphore:
            await _put_single_obstore(store, local_file, remote_key, chunk_size, max_concurrency)

    await asyncio.gather(*(_upload(f) for f in files))
    return to_path


async def put(
    from_path: str,
    to_path: Optional[str] = None,
    recursive: bool = False,
    batch_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Upload a local file or directory to a remote location.

    Args:
        from_path: Local path to upload.
        to_path: Remote destination. If not provided, a random path under the raw data prefix is used.
        recursive: Upload a directory tree rather than a single file.
        batch_size: Number of files uploaded at once for recursive uploads. Defaults to the
            FLYTE_IO_BATCH_SIZE environment variable (default: 32).
        chunk_size: Minimum multipart part size in bytes. Defaults to the FLYTE_IO_UPLOAD_CHUNK_SIZE
            environment variable (default: 5 MiB). Larger files still scale the part size above this to
            stay under the 10,000-part limit; values below 5 MiB are raised to 5 MiB. Applies to S3, GCS
            and Azure; ignored by other filesystems.
        max_concurrency: Number of parts of a single file uploaded at once. Defaults to the
            FLYTE_IO_UPLOAD_MAX_CONCURRENCY environment variable (default: 12). Applies to S3, GCS and
            Azure; ignored by other filesystems.
        kwargs: Additional arguments passed to the underlying filesystem (non-obstore filesystems only).

    Returns:
        str: The remote path the data was uploaded to.
    """
    if not to_path:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        name = pathlib.Path(from_path).name
        to_path = ctx.raw_data.get_random_remote_path(file_name=name)

    if not batch_size:
        batch_size = BATCH_SIZE

    file_system = get_underlying_filesystem(path=to_path)
    from_path = strip_file_header(from_path)

    # Check if we should use obstore bypass (to control multipart part sizing -- see
    # _put_obstore_bypass). Guarded identically to the download bypass in get().
    if _supports_obstore_bypass(file_system):
        return await _put_obstore_bypass(
            from_path,
            to_path,
            recursive,
            batch_size=batch_size,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
        )

    if chunk_size is not None or max_concurrency is not None:
        logger.debug(f"chunk_size / max_concurrency are ignored for {file_system.protocol} uploads")

    if isinstance(file_system, AsyncFileSystem):
        dst = await file_system._put(from_path, to_path, recursive=recursive, batch_size=batch_size, **kwargs)  # pylint: disable=W0212
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

    # The obstore-backed filesystem is duck-typed (callers guard on hasattr for _split_path/_construct_store).
    fs = typing.cast(typing.Any, get_underlying_filesystem(path=path))
    bucket, file_path = fs._split_path(path)  # pylint: disable=W0212
    store: ObjectStore = fs._construct_store(bucket)

    file_handle: AsyncReadableFile | AsyncWritableFile

    if "w" in mode:
        attributes = kwargs.pop("attributes", {})
        max_concurrency = _resolve_max_concurrency(kwargs.pop("max_concurrency", None))
        # The writer uploads one multipart part per buffer, so the buffer size is the part size.
        buffer_size = max(_STREAM_WRITE_BUFFER_SIZE, _UPLOAD_CHUNK_FLOOR)
        if "buffer_size" in kwargs:
            buffer_size = kwargs.pop("buffer_size")
        if "chunk_size" in kwargs:
            buffer_size = kwargs.pop("chunk_size")
        file_handle = obstore.open_writer_async(
            store, file_path, attributes=attributes, buffer_size=buffer_size, max_concurrency=max_concurrency
        )
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
    if _supports_obstore_bypass(fs):
        return await _open_obstore_bypass(path, mode, **kwargs)

    # Fallback to normal open
    if isinstance(fs, AsyncFileSystem):
        return await fs.open_async(path, mode, **kwargs)

    raise OnlyAsyncIOSupportedError(f"Filesystem {fs} does not support async operations")


async def put_stream(
    data_iterable: typing.AsyncIterable[bytes] | bytes,
    *,
    name: str | None = None,
    to_path: str | None = None,
    size_hint: int | None = None,
    chunk_size: int | None = None,
    max_concurrency: int | None = None,
    **kwargs,
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

    Args:
        data_iterable: Iterable of bytes to be streamed.
        name: Name of the file to be created. If not provided, a random name will be generated.
        to_path: Path to the remote location where the data will be stored.
        size_hint: Total number of bytes the stream is expected to yield, if known. On obstore-backed
            filesystems the writer uploads one multipart part per buffer, so a fixed buffer caps a single
            object at ~97.6 GiB (10,000 x 10 MiB). When the hint says the stream is larger than that, the
            part size is scaled up to stay under the limit; smaller streams are unaffected. If
            FLYTE_IO_UPLOAD_CHUNK_SIZE is set above the default buffer, it is used as the part size.
        chunk_size: Minimum multipart part size in bytes, overriding FLYTE_IO_UPLOAD_CHUNK_SIZE for this
            call. Scaled up when `size_hint` requires it; values below 5 MiB are raised to 5 MiB. Applies
            to S3, GCS and Azure; ignored by other filesystems.
        max_concurrency: Number of parts uploaded at once, overriding FLYTE_IO_UPLOAD_MAX_CONCURRENCY
            for this call. Applies to S3, GCS and Azure; ignored by other filesystems.
        kwargs: Additional arguments to be passed to the underlying filesystem.

    Returns:
        str: The path to the remote location where the data was stored.
    """
    if not to_path:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        to_path = ctx.raw_data.get_random_remote_path(file_name=name)

    # Check if we should use obstore bypass
    fs = get_underlying_filesystem(path=to_path)
    try:
        open_kwargs = dict(kwargs)
        if _supports_obstore_bypass(fs):
            if chunk_size is not None:
                open_kwargs["chunk_size"] = _compute_upload_chunk_size(size_hint, floor=chunk_size)
            elif "buffer_size" not in open_kwargs:
                computed = _compute_upload_chunk_size(size_hint)
                if computed > _STREAM_WRITE_BUFFER_SIZE:
                    open_kwargs["chunk_size"] = computed
            if max_concurrency is not None:
                open_kwargs["max_concurrency"] = max_concurrency
        elif chunk_size is not None or max_concurrency is not None:
            logger.debug(f"chunk_size / max_concurrency are ignored for {fs.protocol} uploads")
        file_handle = typing.cast("AsyncWritableFile", await open(to_path, "wb", **open_kwargs))
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

    Args:
        path: Path to the remote location where the data will be downloaded.
        kwargs: Additional arguments to be passed to the underlying filesystem.
        chunk_size: Size of each chunk to be read from the file.

    Returns:
        An async iterator that yields chunks of bytes.
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
    Join multiple paths together. For URIs (e.g. s3://, gs://, abfs://, file://),
    joins using forward slashes regardless of the host operating system to prevent Windows
    backslash corruption. For local filesystem paths, uses os.path.join.

    Args:
        paths: Paths to be joined.
    """
    if not paths:
        return ""
    if len(paths) == 1:
        return str(paths[0])

    first = str(paths[0])
    if is_remote(first) or first.startswith("file://"):
        parsed = urlsplit(first.replace("\\", "/"))
        path_segments = [p.replace("\\", "/").strip("/") for p in paths[1:] if p.replace("\\", "/").strip("/")]

        base_path = parsed.path
        if path_segments:
            all_segments = [base_path.strip("/")] if base_path.strip("/") else []
            all_segments.extend(path_segments)
            new_path = "/" + "/".join(all_segments)
            if str(paths[-1]).endswith(("/", "\\")):
                new_path += "/"
        else:
            new_path = base_path

        return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))

    return str(os.path.join(*paths))


async def exists(path: str, **kwargs) -> bool:
    """
    Check if a path exists.

    Args:
        path: Path to be checked.
        kwargs: Additional arguments to be passed to the underlying filesystem.

    Returns:
        True if the path exists, False otherwise.
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


def get_credentials_error(uri: str, protocol: str) -> str:
    # Check for common credential issues
    if protocol == "s3":
        return (
            f"Failed to download data from {uri}. "
            f"S3 credentials are required to access the data at {uri}. "
            "Please set the following environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )
    elif protocol in ("gs", "gcs"):
        return (
            f"Failed to download data from {uri}. "
            f"GCS credentials are required to access the data at {uri}. "
            f"Please set the following environment variable: GOOGLE_APPLICATION_CREDENTIALS"
        )
    elif protocol in ("abfs", "abfss"):
        return (
            f"Failed to download data from {uri}. "
            f"Azure credentials are required to access the data at {uri}. "
            "Please set the following environment variables: AZURE_STORAGE_ACCOUNT_NAME, "
            "AZURE_STORAGE_ACCOUNT_KEY"
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


register(_OBSTORE_SUPPORTED_PROTOCOLS, asynchronous=True)
fsspec.register_implementation("flyte", FlyteFS, clobber=True)
