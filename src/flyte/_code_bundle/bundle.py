import asyncio
import gzip
import logging
import os
import pathlib
import tempfile
from pathlib import Path
from typing import ClassVar, Type

from async_lru import alru_cache
from flyteidl2.core.tasks_pb2 import TaskTemplate

from flyte._logging import log, logger
from flyte._utils import AsyncLRUCache
from flyte.models import CodeBundle

from ._ignore import GitIgnore, Ignore, StandardIgnore
from ._packaging import create_bundle, list_files_to_bundle, list_relative_files_to_bundle, print_ls_tree
from ._utils import CopyFiles, hash_file

_pickled_file_extension = ".pkl.gz"
_tar_file_extension = ".tar.gz"


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
    o: TaskTemplate,
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

    :return: The code bundle, which contains the path where the code was zipped to.
    """
    logger.debug("Building code bundle.")
    from flyte.remote import upload_file

    if not ignore:
        ignore = (StandardIgnore, GitIgnore)

    logger.debug(f"Finding files to bundle, ignoring as configured by: {ignore}")
    files, digest = list_files_to_bundle(from_dir, True, *ignore, copy_style=copy_style)
    if logger.getEffectiveLevel() <= logging.INFO:
        print_ls_tree(from_dir, files)

    logger.debug("Building code bundle.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path, tar_size, archive_size = create_bundle(from_dir, pathlib.Path(tmp_dir), files, digest)
        logger.info(f"Code bundle created at {bundle_path}, size: {tar_size} MB, archive size: {archive_size} MB")
        if not dryrun:
            hash_digest, remote_path = await upload_file.aio(bundle_path)
            logger.debug(f"Code bundle uploaded to {remote_path}")
        else:
            remote_path = "na"
            if copy_bundle_to:
                import shutil

                # Copy the bundle to the given path
                shutil.copy(bundle_path, copy_bundle_to)
                remote_path = str(copy_bundle_to / bundle_path.name)
            _, hash_digest, _ = hash_file(file_path=bundle_path)
        return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest)


@alru_cache
async def build_code_bundle_from_relative_paths(
    relative_paths: tuple[str, ...],
    from_dir: Path,
    extract_dir: str = ".",
    dryrun: bool = False,
    copy_bundle_to: pathlib.Path | None = None,
) -> CodeBundle:
    """
    Build a code bundle from a list of relative paths.
    :param relative_paths: The list of relative paths to bundle.
    :param from_dir: The directory of the code to bundle. This is the root directory for the source.
    :param ignore: The list of ignores to apply. This is a list of Ignore classes.
    :param extract_dir: The directory to extract the code bundle to, when in the container. It defaults to the current
        working directory.
    :param dryrun: If dryrun is enabled, files will not be uploaded to the control plane.
    :param copy_bundle_to: If set, the bundle will be copied to this path. This is used for testing purposes.
    :param copy_style: What to put into the tarball. (either all, or loaded_modules. if none, skip this function)
    :return: The code bundle, which contains the path where the code was zipped to.
    """
    logger.debug("Building code bundle from relative paths.")
    from flyte.remote import upload_file

    logger.debug("Finding files to bundle")
    files, digest = list_relative_files_to_bundle(relative_paths, from_dir)
    if logger.getEffectiveLevel() <= logging.INFO:
        print_ls_tree(from_dir, files)

    logger.debug("Building code bundle.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path, tar_size, archive_size = create_bundle(from_dir, pathlib.Path(tmp_dir), files, digest)
        logger.info(f"Code bundle created at {bundle_path}, size: {tar_size} MB, archive size: {archive_size} MB")
        if not dryrun:
            hash_digest, remote_path = await upload_file.aio(bundle_path)
            logger.debug(f"Code bundle uploaded to {remote_path}")
        else:
            remote_path = "na"
            if copy_bundle_to:
                import shutil

                # Copy the bundle to the given path
                shutil.copy(bundle_path, copy_bundle_to)
                remote_path = str(copy_bundle_to / bundle_path.name)
            _, hash_digest, _ = hash_file(file_path=bundle_path)
        return CodeBundle(tgz=remote_path, destination=extract_dir, computed_version=hash_digest)


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
    if not dest.is_dir():
        raise ValueError(f"Destination path should be a directory, found {dest}, {dest.stat()}")

    # TODO make storage apis better to accept pathlib.Path
    if bundle.tgz:
        downloaded_bundle = dest / os.path.basename(bundle.tgz)
        if downloaded_bundle.exists():
            return downloaded_bundle.absolute()
        # Download the tgz file
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
