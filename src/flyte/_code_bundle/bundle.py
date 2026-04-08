from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import os
import pathlib
import random
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Type

from async_lru import alru_cache
from flyteidl2.core.tasks_pb2 import TaskTemplate

from flyte._logging import log, logger
from flyte._status import status
from flyte._utils import AsyncLRUCache
from flyte.errors import CodeBundleError
from flyte.models import CodeBundle

from ._ignore import GitIgnore, Ignore, StandardIgnore
from ._packaging import create_bundle, list_files_to_bundle, list_relative_files_to_bundle, print_ls_tree
from ._utils import CopyFiles, hash_file

if TYPE_CHECKING:
    from flyte.app import AppEnvironment

_pickled_file_extension = ".pkl.gz"
_tar_file_extension = ".tar.gz"

_BUNDLE_CACHE_TTL_DAYS = 1


def _scoped_digest(digest: str) -> str:
    """Return a digest scoped to the current endpoint/project/domain."""
    from flyte._persistence._db import _cache_scope

    raw = f"{_cache_scope()}:{digest}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_bundle_cache(digest: str) -> tuple[str, str] | None:
    """Look up a previously uploaded bundle by its file digest. Returns (hash_digest, remote_path) or None."""
    from flyte._persistence._db import LocalDB

    try:
        conn = LocalDB.get_sync()
        cutoff = time.time() - _BUNDLE_CACHE_TTL_DAYS * 86400
        row = conn.execute(
            "SELECT hash_digest, remote_path FROM bundle_cache WHERE digest = ? AND created_at > ?",
            (_scoped_digest(digest), cutoff),
        ).fetchone()
        # Prune expired entries ~5% of the time to avoid doing it on every read
        if random.random() < 0.05:
            with LocalDB._write_lock:
                conn.execute("DELETE FROM bundle_cache WHERE created_at <= ?", (cutoff,))
                conn.commit()
        if row:
            return row[0], row[1]
    except (OSError, sqlite3.Error) as e:
        logger.debug(f"Failed to read bundle cache: {e}")
    return None


def _write_bundle_cache(digest: str, hash_digest: str, remote_path: str) -> None:
    """Persist a successfully uploaded bundle to the SQLite cache."""
    from flyte._persistence._db import LocalDB

    try:
        conn = LocalDB.get_sync()
        with LocalDB._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO bundle_cache (digest, hash_digest, remote_path, created_at) "
                "VALUES (?, ?, ?, ?)",
                (_scoped_digest(digest), hash_digest, remote_path, time.time()),
            )
            conn.commit()
    except (OSError, sqlite3.Error) as e:
        logger.debug(f"Failed to write bundle cache: {e}")


class _PklCache:
    _pkl_cache: ClassVar[AsyncLRUCache[str, str]] = AsyncLRUCache[str, str](maxsize=100)

    @classmethod
    async def put(cls, digest: str, upload_to_path: str, from_path: pathlib.Path) -> str:
        """
        Get the pickled code bundle from the cache or build it if not present.

        :param digest: The hash digest of the task template.
        :param upload_to_path: The path to upload the pickled file to.
        :param from_path: The path to read the pickled file from.
        :return: CodeBundle object containing the pickled file path and the computed version.
        """
        import flyte.storage as storage

        async def put_data() -> str:
            return await storage.put(str(from_path), to_path=str(upload_to_path))

        return await cls._pkl_cache.get(
            key=digest,
            value_func=put_data,
        )


async def build_pkl_bundle(
    o: TaskTemplate | AppEnvironment,
    upload_to_controlplane: bool = True,
    upload_from_dataplane_base_path: str | None = None,
    copy_bundle_to: pathlib.Path | None = None,
) -> CodeBundle:
    """
    Build a Pickled for the given task.

    TODO We can optimize this by having an LRU cache for the function, this is so that if the same task is being
    pickled multiple times, we can avoid the overhead of pickling it multiple times, by copying to a common place
    and reusing based on task hash.

    :param o: Object to be pickled. This is the task template.
    :param upload_to_controlplane: Whether to upload the pickled file to the control plane or not
    :param upload_from_dataplane_base_path: If we are on the dataplane, this is the path where the
        pickled file should be uploaded to. upload_to_controlplane has to be False in this case.
    :param copy_bundle_to: If set, the bundle will be copied to this path. This is used for testing purposes.
    :return: CodeBundle object containing the pickled file path and the computed version.
    """
    import cloudpickle

    if upload_to_controlplane and upload_from_dataplane_base_path:
        raise ValueError("Cannot upload to control plane and upload from dataplane path at the same time.")

    logger.debug("Building pickled code bundle.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = pathlib.Path(tmp_dir) / f"code_bundle{_pickled_file_extension}"
        with gzip.GzipFile(filename=dest, mode="wb", mtime=0) as gzipped:
            cloudpickle.dump(o, gzipped)

        if upload_to_controlplane:
            logger.debug("Uploading pickled code bundle to control plane.")
            from flyte.remote import upload_file

            hash_digest, remote_path = await upload_file.aio(dest)
            return CodeBundle(pkl=remote_path, computed_version=hash_digest)

        elif upload_from_dataplane_base_path:
            from flyte._internal.runtime import io

            _, str_digest, _ = hash_file(file_path=dest)
            upload_path = io.pkl_path(upload_from_dataplane_base_path, str_digest)
            logger.debug(f"Uploading pickled code bundle to dataplane path {upload_path}.")
            final_path = await _PklCache.put(
                digest=str_digest,
                upload_to_path=upload_path,
                from_path=dest,
            )
            return CodeBundle(pkl=final_path, computed_version=str_digest)

        else:
            logger.debug("Dryrun enabled, not uploading pickled code bundle.")
            _, str_digest, _ = hash_file(file_path=dest)
            if copy_bundle_to:
                import shutil

                # Copy the bundle to the given path
                shutil.copy(dest, copy_bundle_to, follow_symlinks=True)
                local_path = copy_bundle_to / dest.name
                return CodeBundle(pkl=str(local_path), computed_version=str_digest)
            return CodeBundle(pkl=str(dest), computed_version=str_digest)


@alru_cache
async def build_code_bundle(
    from_dir: Path,
    *ignore: Type[Ignore],
    extract_dir: str = ".",
    dryrun: bool = False,
    copy_bundle_to: pathlib.Path | None = None,
    copy_style: CopyFiles = "loaded_modules",
    skip_cache: bool = False,
) -> CodeBundle:
    """
    Build the code bundle for the current environment.
    :param from_dir: The directory of the code to bundle. This is the root directory for the source.
    :param extract_dir: The directory to extract the code bundle to, when in the container. It defaults to the current
        working directory.
    :param ignore: The list of ignores to apply. This is a list of Ignore classes.
    :param dryrun: If dryrun is enabled, files will not be uploaded to the control plane.
    :param copy_bundle_to: If set, the bundle will be copied to this path. This is used for testing purposes.
    :param copy_style: What to put into the tarball. (either all, or loaded_modules. if none, skip this function)
    :param skip_cache: If true, skip the persistent SQLite cache lookup and always rebuild/re-upload.

    :return: The code bundle, which contains the path where the code was zipped to.
    """
    if copy_style == "none":
        raise ValueError("If copy_style is 'none', just don't make a code bundle")

    from flyte.remote import upload_file

    if not ignore:
        ignore = (StandardIgnore, GitIgnore)

    logger.debug(f"Finding files to bundle, ignoring as configured by: {ignore}")
    files, digest = list_files_to_bundle(from_dir, True, *ignore, copy_style=copy_style)
    if len(files) == 0:
        raise CodeBundleError(
            f"No files found to bundle in '{from_dir}'.\n"
            "Possible causes:\n"
            "  - The task file is inside a virtual environment directory (e.g., .venv/, venv/)\n"
            "  - The task file is excluded by .gitignore\n"
            "  - The directory does not contain any Python files\n"
            "To debug, check that your task file exists in the specified directory and is not ignored."
        )

    if logger.getEffectiveLevel() <= logging.INFO:
        print_ls_tree(from_dir, files)

    # Check persistent cache before creating the tar bundle to avoid unnecessary work
    if not dryrun and not skip_cache:
        cached = _read_bundle_cache(digest)
        if cached:
            hash_digest, remote_path = cached
            status.success("Code bundle found in cache, skipping upload")
            logger.debug(f"Code bundle cache hit: {remote_path}")
            return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest, files=files)

    status.step("Bundling code...")
    logger.debug("Building code bundle.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path, tar_size, archive_size = create_bundle(
            from_dir, pathlib.Path(tmp_dir), files, digest, deref_symlinks=True
        )
        status.success(f"Code bundle: {len(files)} files, {tar_size} MB (compressed {archive_size} MB)")
        if not dryrun:
            status.step("Uploading code bundle...")
            hash_digest, remote_path = await upload_file.aio(bundle_path)
            logger.debug(f"Code bundle uploaded to {remote_path}")
            _write_bundle_cache(digest, hash_digest, remote_path)
        else:
            if copy_bundle_to:
                remote_path = str(copy_bundle_to / bundle_path.name)
            else:
                import flyte.storage as storage

                base_path = storage.get_random_local_path()
                base_path.mkdir(parents=True, exist_ok=True)
                remote_path = str(base_path / bundle_path.name)

            import shutil

            # Copy the bundle to the given path
            shutil.copy(bundle_path, remote_path)
            _, hash_digest, _ = hash_file(file_path=bundle_path)
        return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest, files=files)


@alru_cache
async def build_code_bundle_from_relative_paths(
    relative_paths: tuple[str, ...],
    from_dir: Path,
    extract_dir: str = ".",
    dryrun: bool = False,
    copy_bundle_to: pathlib.Path | None = None,
    skip_cache: bool = False,
) -> CodeBundle:
    """
    Build a code bundle from a list of relative paths.
    :param relative_paths: The list of relative paths to bundle.
    :param from_dir: The directory of the code to bundle. This is the root directory for the source.
    :param extract_dir: The directory to extract the code bundle to, when in the container. It defaults to the current
        working directory.
    :param dryrun: If dryrun is enabled, files will not be uploaded to the control plane.
    :param copy_bundle_to: If set, the bundle will be copied to this path. This is used for testing purposes.
    :param skip_cache: If true, skip the persistent SQLite cache lookup and always rebuild/re-upload.
    :return: The code bundle, which contains the path where the code was zipped to.
    """
    status.step("Bundling code...")
    logger.debug("Building code bundle from relative paths.")
    from flyte.remote import upload_file

    logger.debug("Finding files to bundle")
    files, digest = list_relative_files_to_bundle(relative_paths, from_dir)
    if logger.getEffectiveLevel() <= logging.INFO:
        print_ls_tree(from_dir, files)

    # Check persistent cache before creating the tar bundle to avoid unnecessary work
    if not dryrun and not skip_cache:
        cached = _read_bundle_cache(digest)
        if cached:
            hash_digest, remote_path = cached
            status.success("Code bundle found in cache, skipping upload")
            logger.debug(f"Code bundle cache hit: {remote_path}")
            return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest, files=files)

    logger.debug("Building code bundle.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path, tar_size, archive_size = create_bundle(from_dir, pathlib.Path(tmp_dir), files, digest)
        status.success(f"Code bundle: {len(files)} files, {tar_size} MB (compressed {archive_size} MB)")
        if not dryrun:
            status.step("Uploading code bundle...")
            hash_digest, remote_path = await upload_file.aio(bundle_path)
            logger.debug(f"Code bundle uploaded to {remote_path}")
            _write_bundle_cache(digest, hash_digest, remote_path)
        else:
            remote_path = "na"
            if copy_bundle_to:
                import shutil

                # Copy the bundle to the given path
                shutil.copy(bundle_path, copy_bundle_to)
                remote_path = str(copy_bundle_to / bundle_path.name)
            _, hash_digest, _ = hash_file(file_path=bundle_path)
        return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest, files=files)


@log(level=logging.INFO)
async def download_bundle(bundle: CodeBundle) -> pathlib.Path:
    """
    Downloads a code bundle (tgz | pkl) to the local destination path.
    :param bundle: The code bundle to download.

    :return: The path to the downloaded code bundle.
    """
    import sys

    import flyte.storage as storage

    dest = pathlib.Path(bundle.destination)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir():
        raise ValueError(f"Destination path should be a directory, found {dest}, {dest.stat()}")

    # TODO make storage apis better to accept pathlib.Path
    if bundle.tgz:
        downloaded_bundle = dest / os.path.basename(bundle.tgz)
        if downloaded_bundle.exists():
            logger.debug(f"Code bundle {downloaded_bundle} already exists locally, skipping download.")
            return downloaded_bundle.absolute()
        # Download the tgz file
        logger.debug(f"Downloading code bundle from {bundle.tgz} to {downloaded_bundle.absolute()}")
        await storage.get(bundle.tgz, str(downloaded_bundle.absolute()))
        # NOTE the os.path.join(destination, ''). This is to ensure that the given path is in fact a directory and all
        # downloaded data should be copied into this directory. We do this to account for a difference in behavior in
        # fsspec, which requires a trailing slash in case of pre-existing directory.
        args = [
            "-xvf",
            str(downloaded_bundle),
            "-C",
            str(dest),
        ]
        if sys.platform != "darwin":
            args.insert(0, "--overwrite")

        process = await asyncio.create_subprocess_exec(
            "tar",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(stderr.decode())
        return downloaded_bundle.absolute()

    elif bundle.pkl:
        # Lets gunzip the pkl file

        downloaded_bundle = dest / os.path.basename(bundle.pkl)
        # Download the tgz file
        await storage.get(bundle.pkl, str(downloaded_bundle.absolute()))
        return downloaded_bundle.absolute()
    else:
        raise ValueError("Code bundle should be either tgz or pkl, found neither.")
