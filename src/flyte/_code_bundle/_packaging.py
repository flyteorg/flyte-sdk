from __future__ import annotations

import gzip
import hashlib
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import tarfile
import time
import typing
from typing import List, Optional, Tuple, Union

import click
from rich.tree import Tree

from flyte._logging import _get_console, logger

from ._ignore import Ignore, IgnoreGroup
from ._utils import (
    CopyFiles,
    _filehash_update,
    _pathhash_update,
    ls_files,
    ls_relative_files,
    tar_strip_file_attributes,
)

FAST_PREFIX = "fast"
FAST_FILEENDING = ".tar.gz"


def print_ls_tree(source: os.PathLike, ls: typing.List[str]):
    logger.info("Files to be copied for fast registration...")

    tree_root = Tree(
        f"File structure:\n:open_file_folder: {source}",
        guide_style="bold bright_blue",
    )
    source_path = pathlib.Path(source).resolve()
    trees = {source_path: tree_root}
    for f in ls:
        fpp = pathlib.Path(f)
        if fpp.parent not in trees:
            # add trees for all intermediate folders
            current = tree_root
            current_path = source_path  # pathlib.Path(source)
            for subdir in fpp.parent.relative_to(source_path).parts:
                current_path = current_path / subdir
                if current_path not in trees:
                    current = current.add(f"{subdir}", guide_style="bold bright_blue")
                    trees[current_path] = current
                else:
                    current = trees[current_path]
        trees[fpp.parent].add(f"{fpp.name}", guide_style="bold bright_blue")

    console = _get_console()
    with console.capture() as capture:
        console.print(tree_root, overflow="ignore", no_wrap=True, crop=False)
    logger.info(f"Root directory: [link=file://{source}]{source}[/link]")
    logger.info(capture.get(), extra={"console": console})


def _compress_tarball(source: pathlib.Path, output: pathlib.Path) -> None:
    """Compress code tarball using pigz if available, otherwise gzip"""
    if pigz := shutil.which("pigz"):
        with open(str(output), "wb") as gzipped:
            subprocess.run([pigz, "--no-time", "-c", str(source)], stdout=gzipped, check=True)
    else:
        start_time = time.time()
        with gzip.GzipFile(filename=str(output), mode="wb", mtime=0) as gzipped:
            with open(source, "rb") as source_file:
                gzipped.write(source_file.read())

        end_time = time.time()
        warning_time = 10
        if end_time - start_time > warning_time:
            click.secho(
                f"Code tarball compression took {end_time - start_time:.0f} seconds. "
                f"Consider installing `pigz` for faster compression.",
                fg="yellow",
            )


def list_files_to_bundle(
    source: pathlib.Path,
    deref_symlinks: bool = False,
    *ignores: typing.Type[Ignore],
    copy_style: CopyFiles = "all",
) -> typing.Tuple[List[str], str]:
    """
    Takes a source directory and returns a list of all files to be included in the code bundle and a hexdigest of the
    included files.
    :param source: The source directory to package
    :param deref_symlinks: Whether to dereference symlinks or not
    :param ignores: A list of Ignore classes to use for ignoring files
    :param copy_style: The copy style to use for the tarball
    :return: A list of all files to be included in the code bundle and a hexdigest of the included files
    """
    ignore = IgnoreGroup(source, *ignores)

    ls, ls_digest = ls_files(source, copy_style, deref_symlinks, ignore)
    logger.debug(f"Hash of files to be included in the code bundle: {ls_digest}")
    return ls, ls_digest


def list_relative_files_to_bundle(
    relative_paths: tuple[str, ...],
    source: pathlib.Path,
) -> typing.Tuple[List[str], str]:
    """
    List the files in the relative paths.

    :param relative_paths: The list of relative paths to bundle.
    :param source: The source directory to package.
    :param ignores: A list of Ignore classes to use for ignoring files
    :param copy_style: The copy style to use for the tarball
    :return: A list of all files to be included in the code bundle and a hexdigest of the included files.
    """
    _source = source

    all_files, digest = ls_relative_files(list(relative_paths), source)
    logger.debug(f"Hash of files to be included in the code bundle: {digest}")
    return all_files, digest


def create_bundle(
    source: pathlib.Path, output_dir: pathlib.Path, ls: List[str], ls_digest: str, deref_symlinks: bool = False
) -> Tuple[pathlib.Path, float, float]:
    """
    Takes a source directory and packages everything not covered by common ignores into a tarball.
    The output_dir is the directory where the tarball and a compressed version of the tarball will be written.
    The output_dir can be a temporary directory.

    :param source: The source directory to package
    :param output_dir: The directory to write the tarball to
    :param deref_symlinks: Whether to dereference symlinks or not
    :param ls: The list of files to include in the tarball
    :param ls_digest: The hexdigest of the included files
    :return: The path to the tarball, the size of the tarball in MB, and the size of the compressed tarball in MB
    """
    # Compute where the archive should be written
    archive_fname = output_dir / f"{FAST_PREFIX}{ls_digest}{FAST_FILEENDING}"
    tar_path = output_dir / "tmp.tar"
    with tarfile.open(str(tar_path), "w", dereference=deref_symlinks) as tar:
        for ws_file in ls:
            rel_path = os.path.relpath(ws_file, start=source)
            tar.add(
                os.path.join(source, ws_file),
                recursive=False,
                arcname=rel_path,
                filter=lambda x: tar_strip_file_attributes(x),
            )

    size_mbs = tar_path.stat().st_size / 1024 / 1024
    _compress_tarball(tar_path, archive_fname)
    asize_mbs = archive_fname.stat().st_size / 1024 / 1024

    return archive_fname, size_mbs, asize_mbs


def compute_digest(source: Union[os.PathLike, List[os.PathLike]], filter: Optional[typing.Callable] = None) -> str:
    """
    Walks the entirety of the source dir to compute a deterministic md5 hex digest of the dir contents.
    :param os.PathLike source:
    :param callable filter:
    :return Text:
    """
    hasher = hashlib.md5()

    def compute_digest_for_file(path: os.PathLike, rel_path: os.PathLike) -> None:
        # Only consider files that exist (e.g. disregard symlinks that point to non-existent files)
        if not os.path.exists(path):
            logger.info(f"Skipping non-existent file {path}")
            return

        # Skip socket files
        if stat.S_ISSOCK(os.stat(path).st_mode):
            logger.info(f"Skip socket file {path}")
            return

        if filter:
            if filter(rel_path):
                return

        _filehash_update(path, hasher)
        _pathhash_update(rel_path, hasher)

    def compute_digest_for_dir(source: os.PathLike) -> None:
        for root, _, files in os.walk(str(source), topdown=True):
            files.sort()

            for fname in files:
                abspath = os.path.join(root, fname)
                relpath = os.path.relpath(abspath, source)
                compute_digest_for_file(pathlib.Path(abspath), pathlib.Path(relpath))

    if isinstance(source, list):
        for src in source:
            if os.path.isdir(src):
                compute_digest_for_dir(src)
            else:
                compute_digest_for_file(src, os.path.basename(src))
    else:
        compute_digest_for_dir(source)

    return hasher.hexdigest()


def get_additional_distribution_loc(remote_location: str, identifier: str) -> str:
    """
    :param Text remote_location:
    :param Text identifier:
    :return Text:
    """
    return posixpath.join(remote_location, "{}.{}".format(identifier, "tar.gz"))
