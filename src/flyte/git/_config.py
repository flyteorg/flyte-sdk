import pathlib
import subprocess
from pathlib import Path

import flyte.config
from flyte._logging import logger


class GitConfig:
    """Configuration and information about the current Git repository."""

    is_valid: bool
    is_tree_clean: bool
    remote_url: str
    repo_dir: Path
    commit_sha: str

    def __init__(self):
        """Initialize all Git-related variables using Git commands.

        If Git is not installed or .git does not exist, marks is_git_repo as False and returns.
        """
        self.is_valid = False
        self.is_tree_clean = False
        self.remote_url = ""
        self.repo_dir = Path.cwd()
        self.commit_sha = ""

        try:
            # Check if we're in a git repository and get the root directory
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.debug("Not in a git repository")
                return

            self.repo_dir = Path(result.stdout.strip())

            # Get current commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.commit_sha = result.stdout.strip()
            else:
                logger.debug("Could not get current git commit SHA")
                return

            # Check if working tree is clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.is_tree_clean = len(result.stdout.strip()) == 0
            else:
                logger.debug("Could not check if working tree is clean")
                return

            # Get remote URL
            self.remote_url = self._get_remote_url()
            if not self.remote_url:
                logger.debug("No remote URL found")
                return
            self.is_valid = True

        except Exception:
            self.is_valid = False

    def _get_remote_url(self) -> str:
        """Get the remote push URL.

        Returns the 'origin' remote push URL if it exists, otherwise returns
        the first remote alphabetically. Removes .git suffix if present.
        """
        try:
            # Try to get origin push remote first
            result = subprocess.run(
                ["git", "remote", "get-url", "--push", "origin"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                url = result.stdout.strip()
                return self._remove_git_suffix(url)

            # If origin doesn't exist, get all remotes
            result = subprocess.run(
                ["git", "remote"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                remotes = result.stdout.strip().split("\n")
                if remotes and remotes[0]:
                    # Sort alphabetically and get the first one
                    remotes.sort()
                    first_remote = remotes[0]

                    # Get push URL for this remote
                    result = subprocess.run(
                        ["git", "remote", "get-url", "--push", first_remote],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        url = result.stdout.strip()
                        return self._remove_git_suffix(url)

            return ""

        except Exception:
            return ""

    def _remove_git_suffix(self, url: str) -> str:
        """Remove .git suffix from URL if present."""
        if url.endswith(".git"):
            return url[:-4]
        return url

    def get_file_path(self, path: Path | str) -> Path:
        """Get the path relative to the repository root directory.

        Args:
            path: Absolute or relative path to a file

        Returns:
            Path relative to repo_dir
        """
        path_obj = Path(path).resolve()
        try:
            return path_obj.relative_to(self.repo_dir)
        except ValueError:
            # Path is not relative to repo_dir, return as-is
            return path_obj


def config_from_root(path: pathlib.Path | str = ".flyte/config.yaml") -> flyte.config.Config | None:
    """Get the config file from the git root directory.

    By default, the config file is expected to be in `.flyte/config.yaml` in the git root directory.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        root = pathlib.Path(result.stdout.strip())
        if not (root / path).exists():
            return None
        return flyte.config.auto(root / path)
    except Exception:
        return None
