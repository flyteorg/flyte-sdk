from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import typing
from pathlib import Path
from typing import List, Optional, Union

from flyte._logging import logger


class _IgnoreLike(typing.Protocol):
    def is_ignored(self, path: pathlib.Path) -> bool: ...


def filehash_update(path: pathlib.Path, hasher: hashlib._Hash) -> None:
    blocksize = 65536
    with open(path, "rb") as f:
        bytes = f.read(blocksize)
        while bytes:
            hasher.update(bytes)
            bytes = f.read(blocksize)


def _pathhash_update(path: Union[os.PathLike, str], hasher: hashlib._Hash) -> None:
    path_list = str(path).split(os.sep)
    hasher.update("".join(path_list).encode("utf-8"))


def update_hasher_for_source(
    source: Union[os.PathLike, List[os.PathLike]],
    hasher: hashlib._Hash,
    ignore: Optional[_IgnoreLike] = None,
):
    """
    Incorporates a single file, or walks a directory tree, into the hasher (content + relative paths).
    :param os.PathLike source:
    :param ignore: Optional ignore instance whose is_ignored(abs_path) determines whether to skip a file.
    :return None:
    """

    def compute_digest_for_file(path: os.PathLike, rel_path: os.PathLike) -> None:
        # Only consider files that exist (e.g. disregard symlinks that point to non-existent files)
        if not os.path.exists(path):
            logger.info(f"Skipping non-existent file {path}")
            return

        # Skip socket files
        if stat.S_ISSOCK(os.stat(path).st_mode):
            logger.info(f"Skip socket file {path}")
            return

        if ignore and ignore.is_ignored(Path(path)):
            return

        filehash_update(Path(path), hasher)
        _pathhash_update(rel_path, hasher)

    def compute_digest_for_dir(source: os.PathLike):
        for root, dirnames, files in os.walk(str(source), topdown=True):
            if ignore:
                dirnames[:] = [d for d in dirnames if not ignore.is_ignored(Path(os.path.join(root, d)))]
            files.sort()

            for fname in files:
                abspath = os.path.join(root, fname)
                relpath = os.path.relpath(abspath, source)
                compute_digest_for_file(Path(abspath), Path(relpath))

    if isinstance(source, list):
        for src in source:
            if os.path.isdir(src):
                compute_digest_for_dir(src)
            else:
                compute_digest_for_file(src, os.path.basename(src))
    elif os.path.isdir(source):
        compute_digest_for_dir(source)
    else:
        compute_digest_for_file(source, os.path.basename(source))
