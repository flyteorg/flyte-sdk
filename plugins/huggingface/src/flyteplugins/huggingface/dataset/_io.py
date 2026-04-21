from __future__ import annotations

import asyncio
import json
import os
import typing

from fsspec.asyn import AsyncFileSystem

import flyte.storage as storage
from flyte._logging import logger

from ._source import (
    HFShard,
    HFSource,
    hf_revision,
    hf_source_cache_key,
    hf_source_payload,
)

_HF_CACHE_MANIFEST = "_flyte_hf_manifest.json"
_HF_DATASET_REGISTRY = "huggingface/datasets"


class HFParquetError(ValueError):
    """Raised when a Hugging Face parquet conversion cannot be resolved."""


def get_hf_cache_path(source: HFSource, cache_key: str | None = None) -> str:
    """Return a deterministic remote-storage path for source."""
    if source.cache_root is None:
        raise ValueError("cache_root is required for deterministic HF cache paths")

    cache_key = cache_key or hf_source_cache_key(source)
    return join_uri_path(source.cache_root, _HF_DATASET_REGISTRY, "blobs", cache_key)


def get_hf_registry_record_path(source: HFSource, cache_key: str) -> str:
    if source.cache_root is None:
        raise ValueError("cache_root is required for HF registry records")

    return join_uri_path(
        source.cache_root,
        _HF_DATASET_REGISTRY,
        "by-key",
        f"{cache_key}.json",
    )


def get_random_hf_path() -> str:
    return os.path.join(str(storage.get_random_local_directory()), "hf-dataset")


def join_uri_path(base: str, *parts: str) -> str:
    """Join URI/object-store path components with POSIX separators."""
    joined = base.rstrip("/")
    for part in parts:
        cleaned = part.strip("/")
        if cleaned:
            joined = f"{joined}/{cleaned}" if joined else cleaned
    return joined


def _storage_kind(path: str) -> str:
    return "remote" if storage.is_remote(path) else "local"


def _source_log_description(source: HFSource) -> str:
    config = source.name if source.name is not None else "auto"
    split = source.split if source.split is not None else "all"
    return f"config={config}, split={split}"


async def run_sync_io(
    label: str,
    func: typing.Callable[..., typing.Any],
    *args: typing.Any,
    **kwargs: typing.Any,
) -> typing.Any:
    """Run blocking sync IO without blocking the active event loop."""

    try:
        # Using to_thread() as the blocking calls are mostly IO-bound.
        return await asyncio.to_thread(func, *args, **kwargs)
    except asyncio.CancelledError:
        logger.warning(
            f"Cancellation requested while running {label}. The active sync IO "
            "call may finish in its worker thread before stopping."
        )
        raise


async def storage_path_exists(path: str) -> bool:
    if storage.is_remote(path):
        try:
            return await storage.exists(path)
        except Exception as e:
            logger.debug(f"Unable to check whether {path} exists: {e}")
            return False

    return typing.cast(bool, await run_sync_io("local exists", os.path.exists, path))


async def storage_read_bytes(path: str) -> bytes:
    if storage.is_remote(path):
        local_path = storage.get_random_local_path(
            file_path_or_file_name=os.path.basename(path)
        )
        await storage.get(path, str(local_path))
        path = str(local_path)

    def _read() -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    return typing.cast(bytes, await run_sync_io("read local file", _read))


async def storage_write_bytes(path: str, data: bytes) -> None:
    if storage.is_remote(path):
        await storage.put_stream(data, to_path=path)
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    def _write() -> None:
        with open(path, "wb") as fh:
            fh.write(data)

    await run_sync_io("write local file", _write)


async def list_parquet_files(uri: str, filesystem) -> list[str]:
    """Return sorted parquet file paths under uri, recursively."""
    try:
        if isinstance(filesystem, AsyncFileSystem):
            raw = sorted(
                f for f in await filesystem._find(uri) if f.endswith(".parquet")
            )
        else:
            found = await run_sync_io("filesystem find parquet", filesystem.find, uri)
            raw = sorted(f for f in found if f.endswith(".parquet"))

        if not raw:
            return [join_uri_path(uri, f"{0:05}.parquet")]

        if "://" in uri and "://" not in raw[0]:
            proto = uri.split("://")[0] + "://"
            raw = [f"{proto}{f}" for f in raw]
        return raw
    except Exception as e:
        logger.debug(f"Unable to list parquet files under {uri}: {e}")
        return [join_uri_path(uri, f"{0:05}.parquet")]


def _format_names(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _list_hf_dir(hfs, path: str, revision: str) -> list[dict[str, typing.Any]]:
    return typing.cast(
        list[dict[str, typing.Any]],
        hfs.ls(path, revision=revision, detail=True),
    )


def _try_list_hf_dir(
    hfs,
    path: str,
    revision: str,
) -> list[dict[str, typing.Any]] | None:
    try:
        return _list_hf_dir(hfs, path, revision)
    except FileNotFoundError:
        return None


def _entry_name(entry: dict[str, typing.Any]) -> str:
    return typing.cast(str, entry["name"]).rstrip("/").split("/")[-1]


def _is_directory(entry: dict[str, typing.Any]) -> bool:
    return entry.get("type") == "directory"


def _is_parquet_file(entry: dict[str, typing.Any]) -> bool:
    return entry.get("type") == "file" and typing.cast(
        str, entry.get("name", "")
    ).endswith(".parquet")


def _config_dirs(entries: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
    return sorted(
        [entry for entry in entries if _is_directory(entry)],
        key=lambda entry: typing.cast(str, entry["name"]),
    )


def _resolve_hf_config(
    hfs,
    source: HFSource,
    revision: str,
) -> tuple[str, str, list[dict[str, typing.Any]]]:
    """Return the config name, path, and entries for source, or raise HFParquetError."""
    repo_path = join_uri_path("datasets", source.repo)

    if source.name is not None:
        config_path = join_uri_path(repo_path, source.name)
        entries = _try_list_hf_dir(hfs, config_path, revision)
        if entries is not None:
            return source.name, config_path, entries

        repo_entries = _try_list_hf_dir(hfs, repo_path, revision) or []
        available = [_entry_name(entry) for entry in _config_dirs(repo_entries)]
        raise HFParquetError(
            f"No Hugging Face parquet config named {source.name!r} found for "
            f"{source.repo} at revision {revision!r}. Available configs: "
            f"{_format_names(available)}."
        )

    default_path = join_uri_path(repo_path, "default")
    default_entries = _try_list_hf_dir(hfs, default_path, revision)
    if default_entries is not None:
        return "default", default_path, default_entries

    repo_entries = _try_list_hf_dir(hfs, repo_path, revision)
    if repo_entries is None:
        raise HFParquetError(
            f"No Hugging Face parquet conversion found for {source.repo} at "
            f"revision {revision!r}."
        )

    configs = _config_dirs(repo_entries)
    config_names = [_entry_name(entry) for entry in configs]
    if len(configs) == 1:
        config_path = typing.cast(str, configs[0]["name"])
        return config_names[0], config_path, _list_hf_dir(hfs, config_path, revision)

    if not configs:
        raise HFParquetError(
            f"No Hugging Face parquet configs found for {source.repo} at "
            f"revision {revision!r}."
        )

    raise HFParquetError(
        f"Hugging Face dataset {source.repo} has multiple parquet configs: "
        f"{_format_names(config_names)}. Pass name=... to from_hf()."
    )


def collect_hf_shards(source: HFSource) -> list[HFShard]:
    """Return parquet shards for a Hugging Face dataset source."""
    import huggingface_hub

    token = os.environ.get("HF_TOKEN")
    if token is None:
        logger.warning(
            "HF_TOKEN not set, using anonymous access. Private datasets will fail."
        )

    hfs = huggingface_hub.HfFileSystem(token=token)
    revision = hf_revision(source)
    config, base_path, base_entries = _resolve_hf_config(hfs, source, revision)

    if source.split:
        split_paths = [(source.split, join_uri_path(base_path, source.split))]
    else:
        split_paths = [
            (_entry_name(entry), typing.cast(str, entry["name"]))
            for entry in base_entries
            if _is_directory(entry)
        ]
        if not split_paths and any(_is_parquet_file(entry) for entry in base_entries):
            split_paths = [("data", base_path)]

    shards: list[HFShard] = []
    for split_name, search_path in split_paths:
        entries = _try_list_hf_dir(hfs, search_path, revision)
        if entries is None:
            raise HFParquetError(
                f"No Hugging Face parquet split named {split_name!r} found for "
                f"{source.repo} (config={config}) at revision {revision!r}."
            )

        for file_info in entries:
            if _is_parquet_file(file_info):
                file_name = file_info["name"].split("/")[-1]
                rel = (
                    file_name if source.split else join_uri_path(split_name, file_name)
                )
                shards.append(
                    HFShard(
                        rel_path=rel,
                        hf_name=file_info["name"],
                        size=file_info.get("size"),
                        etag=file_info.get("etag") or file_info.get("ETag"),
                        last_modified=file_info.get("last_modified"),
                    )
                )

    if not shards:
        raise HFParquetError(
            f"No parquet files found for {source.repo} "
            f"(config={config}, split={source.split}). "
            "The dataset may not have been auto-converted to parquet yet."
        )
    return sorted(shards, key=lambda s: s.rel_path)


def manifest_path(remote_path: str) -> str:
    return join_uri_path(remote_path, _HF_CACHE_MANIFEST)


def hf_cache_manifest(
    source: HFSource,
    shards: list[HFShard],
    cache_key: str,
) -> dict[str, typing.Any]:
    return {
        "version": 1,
        "cache_key": cache_key,
        "source": hf_source_payload(source),
        "shards": hf_source_payload(source, shards)["shards"],
    }


async def read_cache_manifest(
    remote_path: str,
) -> dict[str, typing.Any] | None:
    path = manifest_path(remote_path)
    try:
        if not await storage_path_exists(path):
            return None
        return typing.cast(
            dict[str, typing.Any],
            json.loads((await storage_read_bytes(path)).decode("utf-8")),
        )
    except Exception as e:
        logger.debug(f"Unable to read HF cache manifest {path}: {e}")
        return None


async def write_cache_manifest(
    remote_path: str, manifest: dict[str, typing.Any]
) -> None:
    data = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    await storage_write_bytes(manifest_path(remote_path), data)


async def read_registry_record(
    source: HFSource,
    cache_key: str,
) -> dict[str, typing.Any] | None:
    path = get_hf_registry_record_path(source, cache_key)
    try:
        if not await storage_path_exists(path):
            return None
        return typing.cast(
            dict[str, typing.Any],
            json.loads((await storage_read_bytes(path)).decode("utf-8")),
        )
    except Exception as e:
        logger.debug(f"Unable to read HF registry record {path}: {e}")
        return None


async def write_registry_record(
    source: HFSource,
    cache_key: str,
    artifact_uri: str,
    manifest: dict[str, typing.Any],
) -> None:
    record = {
        **manifest,
        "artifact_uri": artifact_uri,
    }
    data = json.dumps(record, sort_keys=True, indent=2).encode("utf-8")
    await storage_write_bytes(get_hf_registry_record_path(source, cache_key), data)


async def open_sync_hf_reader(hfs, hf_name: str, revision: str):
    return await run_sync_io(
        "open HF shard",
        hfs.open,
        hf_name,
        "rb",
        revision=revision,
    )


async def close_sync_file(label: str, file_obj) -> None:
    close = getattr(file_obj, "close", None)
    if close is not None:
        await run_sync_io(label, close)


async def iter_hf_shard_chunks(
    hfs,
    shard: HFShard,
    *,
    revision: str,
    chunk_size: int,
) -> typing.AsyncIterator[bytes]:
    src = await open_sync_hf_reader(hfs, shard.hf_name, revision)
    try:
        while chunk := await run_sync_io("read HF shard chunk", src.read, chunk_size):
            yield chunk
    finally:
        await close_sync_file("close HF shard", src)


async def stream_hf_shard(
    hfs,
    shard: HFShard,
    dest: str,
    *,
    revision: str,
    chunk_size: int,
) -> None:
    if not storage.is_remote(dest):
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)

    await storage.put_stream(
        iter_hf_shard_chunks(
            hfs,
            shard,
            revision=revision,
            chunk_size=chunk_size,
        ),
        to_path=dest,
    )


async def stream_hf_to_remote(
    source: HFSource,
    remote_path: str,
    shards: list[HFShard] | None = None,
    manifest: dict[str, typing.Any] | None = None,
) -> None:
    """Stream parquet shards from Hugging Face Hub to Flyte remote storage."""
    import huggingface_hub

    token = os.environ.get("HF_TOKEN")
    hfs = huggingface_hub.HfFileSystem(token=token)
    chunk_size = 64 * 1024 * 1024
    revision = hf_revision(source)

    if shards is None:
        shards = typing.cast(
            list[HFShard],
            await run_sync_io("collect HF parquet shards", collect_hf_shards, source),
        )

    logger.info(
        f"Streaming {len(shards)} Hugging Face parquet shard(s) for {source.repo} "
        f"to {_storage_kind(remote_path)} path {remote_path}"
    )
    for shard in shards:
        dest = join_uri_path(remote_path, shard.rel_path)
        logger.info(f"Streaming {shard.rel_path} to {dest}")
        await stream_hf_shard(
            hfs,
            shard,
            dest,
            revision=revision,
            chunk_size=chunk_size,
        )

    if manifest is not None:
        await write_cache_manifest(remote_path, manifest)

    logger.info(f"Streamed {len(shards)} parquet file(s) to {remote_path}")


async def ensure_hf_cached(source: HFSource) -> str:
    """Return the remote path for source, fetching from HF if not cached."""
    shards = typing.cast(
        list[HFShard],
        await run_sync_io("collect HF parquet shards", collect_hf_shards, source),
    )
    cache_key = hf_source_cache_key(source, shards)
    expected_manifest = hf_cache_manifest(source, shards, cache_key)

    if source.cache_root is None:
        remote_path = get_random_hf_path()
        logger.info(
            f"Materializing Hugging Face dataset {source.repo} "
            f"({_source_log_description(source)}) "
            f"to local path {remote_path}"
        )
        await stream_hf_to_remote(source, remote_path, shards, expected_manifest)
        return remote_path

    default_remote_path = get_hf_cache_path(source, cache_key)
    logger.info(
        f"Checking Hugging Face dataset cache for {source.repo} "
        f"({_source_log_description(source)}) "
        f"under {source.cache_root}"
    )
    registry_record = await read_registry_record(source, cache_key)
    remote_path = (
        registry_record.get("artifact_uri", default_remote_path)
        if registry_record is not None
        else default_remote_path
    )

    if await read_cache_manifest(remote_path) == expected_manifest:
        if registry_record is None:
            await write_registry_record(
                source,
                cache_key,
                remote_path,
                expected_manifest,
            )
        logger.info(f"Using cached Hugging Face dataset at {remote_path}")
        return remote_path

    logger.info(
        f"Materializing Hugging Face dataset {source.repo} "
        f"({_source_log_description(source)}) "
        f"to remote cache artifact {remote_path}"
    )
    remote_path = default_remote_path
    await stream_hf_to_remote(source, remote_path, shards, expected_manifest)
    await write_registry_record(
        source,
        cache_key,
        remote_path,
        expected_manifest,
    )
    return remote_path
