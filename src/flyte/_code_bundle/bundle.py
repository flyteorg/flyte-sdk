from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import logging
import os
import pathlib
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, Type

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows local dev; task containers are POSIX
    fcntl = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

from async_lru import alru_cache

from flyte._logging import log, logger
from flyte._status import status
from flyte._utils import AsyncLRUCache
from flyte.errors import CodeBundleError
from flyte.models import CodeBundle

from ._ignore import FlyteIgnore, GitIgnore, Ignore, StandardIgnore
from ._packaging import create_bundle, list_files_to_bundle, list_relative_files_to_bundle, print_ls_tree
from ._utils import CopyFiles, hash_file

if TYPE_CHECKING:
    from flyte._task import TaskTemplate
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
    additional_files: tuple[str, ...] = (),
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
    :param additional_files: Extra absolute paths to bundle in addition to whatever ``copy_style``
        discovers. Used to implement ``Environment.include``. When ``copy_style='none'`` and
        ``additional_files`` is non-empty, falls back to a relative-paths-only bundle.

    :return: The code bundle, which contains the path where the code was zipped to.
    """
    if copy_style == "none":
        if additional_files:
            return await build_code_bundle_from_relative_paths(
                additional_files,
                from_dir=from_dir,
                extract_dir=extract_dir,
                dryrun=dryrun,
                copy_bundle_to=copy_bundle_to,
            )
        raise ValueError("If copy_style is 'none', just don't make a code bundle")

    from flyte.remote import upload_file

    if not ignore:
        # FlyteIgnore applies .flyteignore patterns to *all* files (including git-tracked ones),
        # so large tracked assets (e.g. git-lfs files) can be excluded from the bundle. GitIgnore
        # alone only excludes git-untracked/ignored files, so it never catches tracked files.
        ignore = (StandardIgnore, GitIgnore, FlyteIgnore)

    logger.debug(f"Finding files to bundle, ignoring as configured by: {ignore}")
    files, digest = list_files_to_bundle(
        from_dir, True, *ignore, copy_style=copy_style, additional_files=additional_files or None
    )
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


@contextlib.asynccontextmanager
async def _bundle_lock(lock_path: pathlib.Path) -> AsyncIterator[bool]:
    """
    Exclusive advisory flock on lock_path, yielding True if acquired.

    Yields False when locking is unavailable (no fcntl, or lock file uncreatable, e.g. a read-only
    destination) — callers then fall back to the historical unserialized behavior. flock is released
    automatically by the kernel if the holder dies, so a crashed winner never deadlocks waiters.
    The lock file is deliberately never unlinked: unlink+reopen would hand a late arrival a fresh
    inode whose lock it "wins" while the old inode is still held.
    """
    if fcntl is None:
        logger.debug("File locking unavailable on this platform, proceeding without serialization")
        yield False
        return
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        logger.warning(f"Could not create bundle lock file {lock_path}: {e}, proceeding without serialization")
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(f"Another process is downloading/extracting the code bundle, waiting on {lock_path}")
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield True
    finally:
        os.close(fd)


async def _atomic_download(remote_path: str, target: pathlib.Path) -> None:
    """Download remote_path into target's directory under a temp name, then os.replace into place."""
    import flyte.storage as storage

    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    os.close(fd)
    try:
        logger.debug(f"Downloading code bundle from {remote_path} to {target.absolute()}")
        await storage.get(remote_path, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


async def _extract_tar(archive: pathlib.Path, dest: pathlib.Path) -> None:
    args = [
        "-xvf",
        str(archive),
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


async def _locked_fetch(remote_path: str, dest: pathlib.Path, extract: bool) -> pathlib.Path:
    """
    Download (and optionally extract) remote_path into dest, serialized against concurrent callers.

    Multiple worker processes may run this concurrently against the same destination (e.g. the
    torchrun-spawned ranks of a clustered task each call download_bundle on startup). Exactly one
    of them does the download/extract under an exclusive flock; the rest wait and then observe the
    completion marker. The marker — not the archive's existence — is the "ready" signal, because
    the archive appears on disk before extraction has finished.
    """
    downloaded = dest / os.path.basename(remote_path)
    marker = dest / f".{downloaded.name}{'.extracted' if extract else '.done'}"

    if marker.exists():
        logger.debug(f"Code bundle {downloaded} already downloaded, skipping.")
        return downloaded.absolute()

    async with _bundle_lock(dest / f".{downloaded.name}.lock") as locked:
        if locked and marker.exists():
            # Another process completed the work while we waited on the lock.
            return downloaded.absolute()
        if not locked and downloaded.exists():
            logger.debug(f"Code bundle {downloaded} already exists locally, skipping download.")
            return downloaded.absolute()
        # No marker means the archive (if present) may be a truncated leftover from a killed
        # process — always re-download; os.replace over it is safe.
        await _atomic_download(remote_path, downloaded)
        if extract:
            await _extract_tar(downloaded, dest)
        if locked:
            try:
                marker.touch()
            except OSError as e:
                logger.warning(f"Could not write bundle marker {marker}: {e}")
        return downloaded.absolute()


@log(level=logging.INFO)
async def download_bundle(bundle: CodeBundle) -> pathlib.Path:
    """
    Downloads a code bundle (tgz | pkl) to the local destination path.
    :param bundle: The code bundle to download.

    :return: The path to the downloaded code bundle.
    """
    dest = pathlib.Path(bundle.destination)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir():
        raise ValueError(f"Destination path should be a directory, found {dest}, {dest.stat()}")

    if bundle.tgz:
        return await _locked_fetch(bundle.tgz, dest, extract=True)
    elif bundle.pkl:
        return await _locked_fetch(bundle.pkl, dest, extract=False)
    else:
        raise ValueError("Code bundle should be either tgz or pkl, found neither.")
