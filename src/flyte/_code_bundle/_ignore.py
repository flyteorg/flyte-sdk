import os
import pathlib
import subprocess
import tarfile as _tarfile
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from shutil import which
from functools import cached_property
from typing import List, Optional, Type

from flyte._logging import logger


class Ignore(ABC):
    """Base for Ignores, implements core logic. Children have to implement _is_ignored"""

    def __init__(self, root: Path):
        self.root = root

    def is_ignored(self, path: pathlib.Path) -> bool:
        return self._is_ignored(path)

    def tar_filter(self, tarinfo: _tarfile.TarInfo) -> Optional[_tarfile.TarInfo]:
        if self.is_ignored(pathlib.Path(tarinfo.name)):
            return None
        return tarinfo

    @abstractmethod
    def _is_ignored(self, path: pathlib.Path) -> bool:
        pass


class GitIgnore(Ignore):
    """Uses git cli (if available) to list all ignored files and compare with those."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.has_git = which("git") is not None

    @cached_property
    def git_root(self) -> Optional[Path]:
        return self._get_git_root()

    @cached_property
    def ignore_file_paths(self) -> List[Path]:
        return self._find_ignore_files()

    @cached_property
    def ignored_files(self) -> set[str]:
        return self._list_ignored_files()

    @cached_property
    def ignored_dirs(self) -> set[str]:
        return self._list_ignored_dirs()

    def _get_git_root(self) -> Optional[Path]:
        """Get the git repository root directory"""
        if not self.has_git:
            return None
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            if out.returncode == 0:
                return Path(out.stdout.decode("utf-8").strip())
        except Exception:
            pass
        return None

    def _find_ignore_files(self) -> List[Path]:
        """Find all .gitignore and .flyteignore files in git root, self.root, and subdirectories.
        This runs once during initialization to avoid redundant file system searches.
        Returns files in order: git root files, self.root files, then subdirectory files (sorted by path)."""
        processed_files = []
        seen = set()

        for ignore_file in [".gitignore", ".flyteignore"]:
            # Check git repository root (if different from self.root)
            if self.git_root and self.git_root != self.root:
                git_root_ignore = self.git_root / ignore_file
                if git_root_ignore.exists() and git_root_ignore not in seen:
                    processed_files.append(git_root_ignore)
                    seen.add(git_root_ignore)

            # Check self.root directory
            root_ignore = self.root / ignore_file
            if root_ignore.exists() and root_ignore not in seen:
                processed_files.append(root_ignore)
                seen.add(root_ignore)

        _standard_ignored_dirs = {p for p in STANDARD_IGNORE_PATTERNS if "/" not in p and "*" not in p}

        ignore_names = {".gitignore", ".flyteignore"}
        subdir_ignores = []
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=True):
            # Prune standard-ignored directories — never descend into them
            dirnames[:] = [d for d in dirnames if d not in _standard_ignored_dirs]

            for fname in filenames:
                if fname in ignore_names:
                    p = Path(dirpath) / fname
                    if p not in seen:
                        subdir_ignores.append(p)
                        seen.add(p)

        processed_files.extend(subdir_ignores)

        # Log all ignore files being used
        if processed_files:
            ignore_files_list = [str(f) for f in processed_files]
            logger.debug(f"Using ignore files: {', '.join(ignore_files_list)}")

        return processed_files

    def _git_wrapper(self, extra_args: List[str]) -> set[str]:
        # Use absolute paths for all --exclude-from arguments to avoid path resolution issues
        for ignore_file_path in self.ignore_file_paths:
            extra_args.extend([f"--exclude-from={ignore_file_path.absolute()}"])

        if self.has_git:
            out = subprocess.run(
                ["git", "ls-files", "-io", *extra_args],
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            if out.returncode == 0:
                return set(out.stdout.decode("utf-8").split("\n")[:-1])
            logger.info(f"Could not determine ignored paths due to:\n{out.stderr!r}\nNot applying any filters")
            return set()
        logger.info("No git executable found, not applying any filters")
        return set()

    def _list_ignored_files(self) -> set[str]:
        return self._git_wrapper([])

    def _list_ignored_dirs(self) -> set[str]:
        return self._git_wrapper(["--directory"])

    def _is_ignored(self, path: pathlib.Path) -> bool:
        if self.ignored_files:
            # Convert absolute path to relative path for comparison with git output
            try:
                rel_path = path.relative_to(self.root)
            except ValueError:
                # If path is not under root, don't ignore it
                return False

            # git-ls-files uses POSIX paths
            if rel_path.as_posix() in self.ignored_files:
                return True
            # Ignore empty directories
            if os.path.isdir(os.path.join(self.root, path)) and self.ignored_dirs:
                return rel_path.as_posix() + "/" in self.ignored_dirs
        return False


STANDARD_IGNORE_PATTERNS = [
    ".git",
    "*.pyc",
    "**/*.pyc",
    "__pycache__",
    "**/__pycache__",
    ".cache",
    ".cache/*",
    ".ruff_cache",
    "**/.ruff_cache",
    ".mypy_cache",
    "**/.mypy_cache",
    ".pytest_cache",
    "**/.pytest_cache",
    ".venv",
    "**/.venv",
    ".idea",
    "**/.idea",
    "venv",
    "env",
    "*.log",
    ".env",
    "*.egg-info",
    "**/*.egg-info",
    "*.egg",
    "dist",
    "build",
    "*.whl",
]


class StandardIgnore(Ignore):
    """Retains the standard ignore functionality that previously existed. Could in theory
    by fed with custom ignore patterns from cli."""

    def __init__(self, root: Path, patterns: Optional[List[str]] = None):
        super().__init__(root.resolve())
        self.patterns = patterns or STANDARD_IGNORE_PATTERNS

    def _is_ignored(self, path: pathlib.Path) -> bool:
        # Convert to relative path for pattern matching
        try:
            rel_path = path.relative_to(self.root)
        except ValueError:
            # If path is not under root, don't ignore it
            return False

        for pattern in self.patterns:
            if fnmatch(str(rel_path), pattern):
                return True
        return False


class IgnoreGroup(Ignore):
    """Groups multiple Ignores and checks a path against them. A file is ignored if any
    Ignore considers it ignored."""

    def __init__(self, root: Path, *ignores: Type[Ignore]):
        super().__init__(root)
        self.ignores = [ignore(root) for ignore in ignores]

    def _is_ignored(self, path: pathlib.Path) -> bool:
        for ignore in self.ignores:
            if ignore.is_ignored(path):
                return True
        return False

    def list_ignored(self) -> List[str]:
        ignored = []
        for dir, _, files in os.walk(self.root):
            dir_path = Path(dir)
            for file in files:
                abs_path = dir_path / file
                if self.is_ignored(abs_path):
                    ignored.append(str(abs_path.relative_to(self.root)))
        return ignored
