import os
import pathlib
import subprocess
import tarfile as _tarfile
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from shutil import which
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
        self.ignored_files = self._list_ignored_files()
        self.ignored_dirs = self._list_ignored_dirs()

    def _git_wrapper(self, extra_args: List[str]) -> set[str]:
        if self.has_git:
            out = subprocess.run(
                ["git", "ls-files", "-io", "--exclude-standard", *extra_args],
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
            # git-ls-files uses POSIX paths
            if Path(path).as_posix() in self.ignored_files:
                return True
            # Ignore empty directories
            if os.path.isdir(os.path.join(self.root, path)) and self.ignored_dirs:
                return Path(path).as_posix() + "/" in self.ignored_dirs
        return False


STANDARD_IGNORE_PATTERNS = ["*.pyc", ".cache", ".cache/*", "__pycache__", "**/__pycache__"]


class StandardIgnore(Ignore):
    """Retains the standard ignore functionality that previously existed. Could in theory
    by fed with custom ignore patterns from cli."""

    def __init__(self, root: Path, patterns: Optional[List[str]] = None):
        super().__init__(root.resolve())
        self.patterns = patterns if patterns else STANDARD_IGNORE_PATTERNS

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
