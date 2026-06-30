import os
import pathlib
import subprocess
import tarfile as _tarfile
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from shutil import which
from typing import Any, List, Optional, Type

from flyte._logging import logger


def _get_git_root(root: Path) -> Optional[Path]:
    """Return the git repository root for the given directory, or None if not in a repo."""
    if not which("git"):
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if out.returncode == 0:
            return Path(out.stdout.decode("utf-8").strip())
    except Exception:
        pass
    return None


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
        return _get_git_root(self.root)

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
        return _get_git_root(self.root)

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


class DockerfileIgnore(Ignore):
    """Ignores files matching patterns in a .dockerignore file in the given root directory."""

    def __init__(self, root: Path):
        super().__init__(root)
        self._matcher = self._load_matcher()

    def _load_matcher(self):
        dockerignore = self.root / ".dockerignore"
        if not dockerignore.exists():
            return None
        patterns: List[str] = []
        try:
            with open(dockerignore, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        patterns.append(stripped)
        except Exception as e:
            logger.warning(f"Failed to read .dockerignore at {dockerignore}: {e}")
            return None
        from flyte._internal.imagebuild.docker import PatternMatcher

        return PatternMatcher(patterns) if patterns else None

    def _is_ignored(self, path: pathlib.Path) -> bool:
        if not self._matcher:
            return False
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False
        return self._matcher.matches(str(rel))


def _normalize_flyteignore_pattern(pattern: str) -> List[str]:
    """Convert a single .gitignore-style pattern into one or more patterns that
    docker's ``PatternMatcher`` (which uses anchored .dockerignore semantics)
    can interpret with the same effective behavior as git.

    Gitignore rules we emulate:
    - A pattern with no internal slash (e.g. ``*.csv``, ``secrets.json``,
      ``data/``) matches at any depth — emit both the bare form (for the
      top level) and a ``**/`` prefixed form (for nested directories).
    - A pattern beginning with ``/`` is anchored to the directory of the
      .flyteignore file — strip the leading slash.
    - A pattern containing an internal slash (e.g. ``src/foo.py``) is
      already anchored — pass through unchanged.
    - A trailing ``/`` marks a directory pattern — emit a ``/**`` suffix
      form so the directory's contents are also excluded.
    - Negation (``!pattern``) is preserved across the expansion.
    """
    negation = pattern.startswith("!")
    body = pattern[1:] if negation else pattern
    sign = "!" if negation else ""

    # Leading slash → anchored to the .flyteignore's directory; strip it.
    if body.startswith("/"):
        body = body[1:]
        if body.endswith("/"):
            return [sign + body, sign + body + "**"]
        return [sign + body]

    stripped = body.rstrip("/")
    is_dir = body.endswith("/")

    if "/" in stripped:
        # Contains an internal slash → anchored (gitignore semantics).
        if is_dir:
            return [sign + body, sign + body + "**"]
        return [sign + body]

    # Bare pattern → match at any depth.
    if is_dir:
        return [sign + body + "**", sign + "**/" + body + "**"]
    return [sign + body, sign + "**/" + body]


class FlyteIgnore(Ignore):
    """Reads .flyteignore files and excludes matching files from the bundle,
    regardless of whether those files are tracked in git or not.

    This complements GitIgnore: while GitIgnore only excludes files that are
    untracked/ignored by git, FlyteIgnore applies .flyteignore patterns to all
    files — so tracked (committed) files can be excluded from bundles too.

    Patterns use .gitignore syntax (bare patterns match at any depth, leading
    ``/`` anchors to the .flyteignore's directory, trailing ``/`` marks a
    directory, ``!`` negates).
    """

    def __init__(self, root: Path):
        super().__init__(root)

    @cached_property
    def _rules(self) -> List[tuple[Path, Any]]:
        from flyte._internal.imagebuild.docker import PatternMatcher

        rules = []
        for flyteignore_path in self._find_flyteignore_files():
            try:
                lines = flyteignore_path.read_text(encoding="utf-8").splitlines()
            except Exception as e:
                logger.warning(f"Failed to read {flyteignore_path}: {e}")
                continue
            raw = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
            expanded: List[str] = []
            for p in raw:
                expanded.extend(_normalize_flyteignore_pattern(p))
            if expanded:
                rules.append((flyteignore_path.parent, PatternMatcher(expanded)))
        return rules

    def _find_flyteignore_files(self) -> List[Path]:
        """Find all .flyteignore files under root, plus any at the git repo root.

        Order: git root (if outside self.root) → self.root → subdirectories.
        Standard-ignored directories (e.g. .venv, __pycache__) are skipped."""
        seen: set[Path] = set()
        result: List[Path] = []

        git_root = _get_git_root(self.root)
        if git_root and git_root != self.root:
            candidate = git_root / ".flyteignore"
            if candidate.exists():
                result.append(candidate)
                seen.add(candidate)

        root_candidate = self.root / ".flyteignore"
        if root_candidate.exists() and root_candidate not in seen:
            result.append(root_candidate)
            seen.add(root_candidate)

        _standard_ignored_dirs = {p for p in STANDARD_IGNORE_PATTERNS if "/" not in p and "*" not in p}
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in _standard_ignored_dirs]
            for fname in filenames:
                if fname == ".flyteignore":
                    p = Path(dirpath) / fname
                    if p not in seen:
                        result.append(p)
                        seen.add(p)

        return result

    def _is_ignored(self, path: pathlib.Path) -> bool:
        for directory, matcher in self._rules:
            try:
                rel = path.relative_to(directory)
            except ValueError:
                continue
            if matcher.matches(rel.as_posix()):
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
