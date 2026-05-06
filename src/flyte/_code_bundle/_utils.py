from __future__ import annotations

import glob
import gzip
import hashlib
import importlib.util
import os
import pathlib
import shutil
import site
import stat
import sys
import tarfile
import tempfile
import typing
from datetime import datetime, timezone
from functools import lru_cache
from types import ModuleType
from typing import List, Literal, Optional, Sequence, Tuple, Union

from flyte._logging import logger

from ._ignore import Ignore, IgnoreGroup, StandardIgnore

CopyFiles = Literal["loaded_modules", "all", "none", "custom"]


def compress_scripts(source_path: str, destination: str, modules: List[ModuleType]):
    """
    Compresses the single script while maintaining the folder structure for that file.

    For example, given the follow file structure:
    .
    ├── flyte
              ├── __init__.py
              └── workflows
                  ├── example.py
                  ├── another_example.py
                  ├── yet_another_example.py
                  ├── unused_example.py
                  └── __init__.py

    Let's say you want to compress `example.py` imports `another_example.py`. And `another_example.py`
    imports on `yet_another_example.py`. This will  produce a tar file that contains only that
    file alongside with the folder structure, i.e.:

    .
    ├── flyte
              ├── __init__.py
              └── workflows
                  ├── example.py
                  ├── another_example.py
                  ├── yet_another_example.py
                  └── __init__.py

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        destination_path = os.path.join(tmp_dir, "code")
        os.mkdir(destination_path)
        add_imported_modules_from_source(source_path, destination_path, modules)

        tar_path = os.path.join(tmp_dir, "tmp.tar")
        with tarfile.open(tar_path, "w") as tar:
            tmp_path: str = os.path.join(tmp_dir, "code")
            files: typing.List[str] = os.listdir(tmp_path)
            for ws_file in files:
                tar.add(os.path.join(tmp_path, ws_file), arcname=ws_file, filter=tar_strip_file_attributes)
        with gzip.GzipFile(filename=destination, mode="wb", mtime=0) as gzipped:
            with open(tar_path, "rb") as tar_file:
                gzipped.write(tar_file.read())


# Takes in a TarInfo and returns the modified TarInfo:
# https://docs.python.org/3/library/tarfile.html#tarinfo-objects
# intended to be passed as a filter to tarfile.add
# https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.add
def tar_strip_file_attributes(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    # set time to epoch timestamp 0, aka 00:00:00 UTC on 1 January 1981
    # note that when extracting this tarfile, this time will be shown as the modified date
    tar_info.mtime = datetime(1981, 1, 1, tzinfo=timezone.utc).timestamp()

    # user/group info
    tar_info.uid = 0
    tar_info.uname = ""
    tar_info.gid = 0
    tar_info.gname = ""

    # stripping paxheaders may not be required
    # see https://stackoverflow.com/questions/34688392/paxheaders-in-tarball
    tar_info.pax_headers = {}

    return tar_info


def ls_files(
    source_path: pathlib.Path,
    copy_file_detection: CopyFiles,
    deref_symlinks: bool = False,
    ignore_group: Optional[IgnoreGroup] = None,
    additional_files: Optional[Sequence[str]] = None,
) -> Tuple[List[str], str]:
    """
    user_modules_and_packages is a list of the Python modules and packages, expressed as absolute paths, that the
    user has run this command with. For flyte run for instance, this is just a list of one.
    This is used for two reasons.
      - Everything in this list needs to be returned. Files are returned and folders are walked.
      - A common source path is derived from this is, which is just the common folder that contains everything in the
        list. For ex. if you do
        $ pyflyte --pkgs a.b,a.c package
        Then the common root is just the folder a/. The modules list is filtered against this root. Only files
        representing modules under this root are included

    If the copy enum is set to loaded_modules, then the loaded sys modules will be used.

    :param additional_files: Absolute paths that must be included in addition to the files
        discovered via ``copy_file_detection``. Each path must be under ``source_path`` and
        may be a file, a directory (recursively included), or a glob pattern. Used to
        implement ``Environment.include`` across bundling strategies.
    """

    # Unlike the below, the value error here is useful and should be returned to the user, like if absolute and
    # relative paths are mixed.

    # This is --copy auto
    if copy_file_detection == "loaded_modules":
        sys_modules = list(sys.modules.values())
        all_files = list_imported_modules_as_files(str(source_path), sys_modules)
    # this is --copy all (--copy none should never invoke this function)
    else:
        all_files = list_all_files(source_path, deref_symlinks, ignore_group)

    if additional_files:
        resolved_source = source_path.resolve()
        extra_paths: list[str] = []
        for entry in additional_files:
            p = pathlib.Path(entry)
            if p.is_dir():
                extra_paths.extend(str(child) for child in p.glob("**/*") if child.is_file())
            elif p.is_file():
                extra_paths.append(str(p))
            else:
                matched = glob.glob(str(p))
                if not matched:
                    raise ValueError(f"include path {entry!r} is not a file, directory, or matching glob pattern.")
                extra_paths.extend(m for m in matched if pathlib.Path(m).is_file())

        existing = set(all_files)
        for extra in extra_paths:
            # Verify containment on resolved paths (handles macOS /private symlinks, ..
            # segments, etc.) but append the path in a form that is a literal subpath
            # of `source_path`, so the downstream `relative_to(source_path)` succeeds.
            resolved = pathlib.Path(extra).resolve()
            try:
                rel = resolved.relative_to(resolved_source)
            except ValueError as exc:
                raise ValueError(
                    f"include path {extra!r} is outside the bundle root {source_path!s}. "
                    f"Pass --root-dir (or configure it) one level up so every include lives under the root."
                ) from exc
            normalized = str(source_path / rel)
            if normalized not in existing:
                existing.add(normalized)
                all_files.append(normalized)

    all_files.sort()
    hasher = hashlib.md5()
    for abspath in all_files:
        # Use POSIX-style path for hashing to ensure consistent hashes across platforms
        relpath = pathlib.Path(abspath).relative_to(source_path).as_posix()
        _filehash_update(abspath, hasher)
        _pathhash_update(relpath, hasher)

    digest = hasher.hexdigest()

    return all_files, digest


def ls_relative_files(relative_paths: list[str], source_path: pathlib.Path) -> tuple[list[str], str]:
    relative_paths = list(relative_paths)
    relative_paths.sort()
    hasher = hashlib.md5()

    all_files: list[str] = []
    for file in relative_paths:
        path = source_path / file
        if path.is_dir():
            # Filter out directories, only include files
            all_files.extend([str(p) for p in path.glob("**/*") if p.is_file()])
        elif path.is_file():
            all_files.append(str(path))
        else:
            glob_files = glob.glob(str(path))
            if glob_files:
                # Filter out directories from glob results
                all_files.extend([str(f) for f in glob_files if pathlib.Path(f).is_file()])
            else:
                raise ValueError(f"File {path} is not a valid file, directory, or glob pattern")

    all_files.sort()
    resolved_source = source_path.resolve()
    for p in all_files:
        _filehash_update(p, hasher)
        # Resolve before relative_to to normalize any ".." in the path — un-normalized
        # paths would produce inconsistent hashes across equivalent paths.
        rel_path = pathlib.Path(p).resolve().relative_to(resolved_source).as_posix()
        _pathhash_update(rel_path, hasher)

    digest = hasher.hexdigest()
    return all_files, digest


def _filehash_update(path: Union[os.PathLike, str], hasher: hashlib._Hash) -> None:
    blocksize = 65536
    with open(path, "rb") as f:
        chunk = f.read(blocksize)
        while chunk:
            hasher.update(chunk)
            chunk = f.read(blocksize)


def _pathhash_update(path: Union[os.PathLike, str], hasher: hashlib._Hash) -> None:
    path_list = str(path).split(os.sep)
    hasher.update("".join(path_list).encode("utf-8"))


EXCLUDE_DIRS = {".git"}


def list_all_files(source_path: pathlib.Path, deref_symlinks, ignore_group: Optional[Ignore] = None) -> List[str]:
    all_files = []
    source_path_str = str(source_path.absolute())

    # This is needed to prevent infinite recursion when walking with followlinks
    visited_inodes = set()
    for root, dirnames, files in os.walk(source_path_str, topdown=True, followlinks=deref_symlinks):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        # Filter out ignored directories to avoid walking into them
        if ignore_group:
            dirnames[:] = [d for d in dirnames if not ignore_group.is_ignored(pathlib.Path(os.path.join(root, d)))]

        if deref_symlinks:
            inode = os.stat(root).st_ino
            if inode in visited_inodes:
                continue
            visited_inodes.add(inode)

        ff = []
        files.sort()
        for fname in files:
            abspath = os.path.join(root, fname)
            # Only consider files that exist (e.g. disregard symlinks that point to non-existent files)
            if not os.path.exists(abspath):
                logger.info(f"Skipping non-existent file {abspath}")
                continue
            # Skip socket files
            if stat.S_ISSOCK(os.stat(abspath).st_mode):
                logger.info(f"Skip socket file {abspath}")
                continue
            if ignore_group:
                if ignore_group.is_ignored(pathlib.Path(abspath)):
                    continue

            ff.append(abspath)
        all_files.extend(ff)

        # Remove directories that we've already visited from dirnames
        if deref_symlinks:
            dirnames[:] = [d for d in dirnames if os.stat(os.path.join(root, d)).st_ino not in visited_inodes]

    return all_files


def _file_is_in_directory(file: str, directory: str) -> bool:
    """Return True if file is in directory and in its children."""
    try:
        return pathlib.Path(file).resolve().is_relative_to(pathlib.Path(directory).resolve())
    except OSError as e:
        # OSError can be raised if paths cannot be resolved (permissions, broken symlinks, etc.)
        logger.debug(f"Failed to resolve paths for {file} and {directory}: {e!s}")
        return False


def list_imported_modules_as_files(source_path: str, modules: List[ModuleType]) -> List[str]:
    """Lists the files of modules that have been loaded.  The files are only included if:

    1. Not a site-packages. These are installed packages and not user files.
    2. Not in the sys.base_prefix or sys.prefix. These are also installed and not user files.
    3. Shares a common path with the source_path.
    """

    import flyte
    from flyte._utils.lazy_module import is_imported

    files = set()
    flyte_root = os.path.dirname(flyte.__file__)

    # These directories contain installed packages or modules from the Python standard library.
    # If a module is from these directories, then they are not user files.
    invalid_directories = [flyte_root, sys.prefix, sys.base_prefix, site.getusersitepackages(), *site.getsitepackages()]

    for mod in modules:
        # Be careful not to import a module with the .__file__ call if not yet imported.
        if "LazyModule" in object.__getattribute__(mod, "__class__").__name__:
            name = object.__getattribute__(mod, "__name__")
            if is_imported(name):
                mod_file = mod.__file__
            else:
                continue
        else:
            try:
                mod_file = mod.__file__
            except AttributeError:
                continue

        # skip if mod_file is (a) None or (b) not a string. (b) can happen if a third-party package overrides
        # sys.modules[mod.__name__] with a custom object.
        if mod_file is None or not isinstance(mod_file, str):
            continue

        if any(_file_is_in_directory(mod_file, directory) for directory in invalid_directories):
            continue

        if not _file_is_in_directory(mod_file, source_path):
            # Only upload files where the module file in the source directory
            # print log line for files that have common ancestor with source_path, but not in it.
            logger.debug(f"{mod_file} is not in {source_path}")
            continue

        if not pathlib.Path(mod_file).is_file():
            # Some modules have a __file__ attribute that are relative to the base package. Let's skip these,
            # can add more rigorous logic to really pull out the correct file location if we need to.
            logger.debug(f"Skipping {mod_file} from {mod.__name__} because it is not a file")
            continue

        files.add(mod_file)

    return list(files)


def add_imported_modules_from_source(source_path: str, destination: str, modules: List[ModuleType]):
    """Copies modules into destination that are in modules. The module files are copied only if:

    1. Not a site-packages. These are installed packages and not user files.
    2. Not in the sys.base_prefix or sys.prefix. These are also installed and not user files.
    3. Does not share a common path with the source_path.
    """
    # source path is the folder holding the main script.
    # but in register/package case, there are multiple folders.
    # identify a common root amongst the packages listed?

    files = list_imported_modules_as_files(source_path, modules)
    for file in files:
        relative_path = os.path.relpath(file, start=source_path)
        new_destination = os.path.join(destination, relative_path)

        if os.path.exists(new_destination):
            # No need to copy if it already exists
            continue

        os.makedirs(os.path.dirname(new_destination), exist_ok=True)
        shutil.copy(file, new_destination)


def copy_code_bundle_to_context(
    root_dir: pathlib.Path,
    copy_style: CopyFiles,
    context_path: pathlib.Path,
    ignore_patterns: Optional[List[str]] = None,
) -> pathlib.Path:
    """Copy source files for a CodeBundleLayer into a build context directory.

    :param root_dir: The root directory to copy files from.
    :param copy_style: "loaded_modules" to copy only imported modules, "all" to copy everything.
    :param context_path: The build context directory.
    :param ignore_patterns: Ignore patterns for the "all" case.  When *None* the
        `STANDARD_IGNORE_PATTERNS` are used.
    :return: The path within context_path where files were copied.
    """
    resolved_root = root_dir.resolve()

    # Determine destination path (absolute roots go under _flyte_abs_context)
    if root_dir.is_absolute():
        rel_path = pathlib.PurePath(*root_dir.parts[1:])
        dst_path = context_path / "_flyte_abs_context" / rel_path
    else:
        dst_path = context_path / root_dir

    # Reuse ls_files to list files (handles both "loaded_modules" and "all")
    ignore = IgnoreGroup(resolved_root, StandardIgnore)
    if ignore_patterns is not None:
        ignore.ignores = [StandardIgnore(resolved_root, ignore_patterns)]
    all_files, _ = ls_files(resolved_root, copy_style, deref_symlinks=False, ignore_group=ignore)

    # Copy listed files into the context
    for abs_path in all_files:
        rel = pathlib.Path(abs_path).relative_to(resolved_root)
        dest = dst_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, dest)

    return dst_path


def import_module_from_file(module_name, file):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        return module
    except Exception as exc:
        raise ModuleNotFoundError(f"Module from file {file} cannot be loaded") from exc


def get_all_modules(source_path: str, module_name: Optional[str]) -> List[ModuleType]:
    """Import python file with module_name in source_path and return all modules."""
    sys_modules = list(sys.modules.values())
    if module_name is None or module_name in sys.modules:
        # module already exists, there is no need to import it again
        return sys_modules

    full_module = os.path.join(source_path, *module_name.split("."))
    full_module_path = f"{full_module}.py"

    is_python_file = os.path.exists(full_module_path) and os.path.isfile(full_module_path)
    if not is_python_file:
        return sys_modules

    try:
        new_module = import_module_from_file(module_name, full_module_path)
        return [*sys_modules, new_module]
    except Exception as exc:
        logger.error(f"Using system modules, failed to import {module_name} from {full_module_path}: {exc!s}")
        # Import failed so we fallback to `sys_modules`
        return sys_modules


@lru_cache
def hash_file(file_path: typing.Union[os.PathLike, str]) -> Tuple[bytes, str, int]:
    """
    Hash a file and produce a digest to be used as a version
    """
    h = hashlib.md5()
    size = 0

    with open(file_path, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)

    return h.digest(), h.hexdigest(), size
