from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import typing
from pathlib import Path
from typing import List, Optional, Union

from flyte._logging import logger


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
    source: Union[os.PathLike, List[os.PathLike]], hasher: hashlib._Hash, filter: Optional[typing.Callable] = None
):
    """
    Walks the entirety of the source dir to compute a deterministic md5 hex digest of the dir contents.
    :param os.PathLike source:
    :param callable filter:
    :return Text:
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

        if filter:
            if filter(rel_path):
                return

        filehash_update(Path(path), hasher)
        _pathhash_update(rel_path, hasher)

    def compute_digest_for_dir(source: os.PathLike):
        for root, _, files in os.walk(str(source), topdown=True):
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
    else:
        compute_digest_for_dir(source)
