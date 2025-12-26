import pathlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

import flyte.config
from flyte._logging import logger


class GitUrlBuilder(Protocol):
    @staticmethod
    def build_url(remote_url: str, file_path: str, commit_sha: str, line_number: int, is_tree_clean: bool) -> str: ...


class GithubUrlBuilder(GitUrlBuilder):
    host_name = "github.com"

    @staticmethod
    def build_url(remote_url: str, file_path: str, commit_sha: str, line_number: int, is_tree_clean: bool) -> str:
        url = f"{remote_url}/blob/{commit_sha}/{file_path}"
        if is_tree_clean:
            url += f"#L{line_number}"
        return url


class GitlabUrlBuilder(GitUrlBuilder):
    host_name = "gitlab.com"

    @staticmethod
    def build_url(remote_url: str, file_path: str, commit_sha: str, line_number: int, is_tree_clean: bool) -> str:
        url = f"{remote_url}/-/blob/{commit_sha}/{file_path}"
        if is_tree_clean:
            url += f"#L{line_number}"
        return url


GIT_URL_BUILDER_REGISTRY: Dict[str, GitUrlBuilder] = {
    GithubUrlBuilder.host_name: GithubUrlBuilder,
    GitlabUrlBuilder.host_name: GitlabUrlBuilder,
}


@dataclass(init=True, frozen=True)
class GitStatus:
    """
    A class representing the status of a git repository.

    :param is_valid: Whether git repository is valid
    :param is_tree_clean: Whether working tree is clean
    :param remote_url: Remote URL in HTTPS format
    :param repo_dir: Repository root directory
    :param commit_sha: Current commit SHA
    """

    is_valid: bool = False
    is_tree_clean: bool = False
    remote_url: str = ""
    repo_dir: Path = Path()
    commit_sha: str = ""

    @classmethod
    def from_current_repo(cls) -> "GitStatus":
        """Discover git information from the current repository.

        If Git is not installed or .git does not exist, returns GitStatus with is_valid=False.

        :return: GitStatus instance with discovered git information
        """
        try:
            # Check if we're in a git repository and get the root directory
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning("Not in a git repository or git is not installed")
                return cls()

            repo_dir = Path(result.stdout.strip())

            # Get current commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                commit_sha = result.stdout.strip()
            else:
                logger.warning("Failed to get current commit SHA")
                return cls(repo_dir=repo_dir)

            # Check if working tree is clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                is_tree_clean = len(result.stdout.strip()) == 0
            else:
                logger.warning("Failed to check if working tree is clean")
                return cls(repo_dir=repo_dir, commit_sha=commit_sha)

            # Get remote URL
            instance = cls(repo_dir=repo_dir, commit_sha=commit_sha, is_tree_clean=is_tree_clean)
            remote_url = instance._get_remote_url()
            if not remote_url:
                logger.warning("Failed to get remote URL")
                return cls(repo_dir=repo_dir, commit_sha=commit_sha, is_tree_clean=is_tree_clean)

            return cls(
                is_valid=True,
                is_tree_clean=is_tree_clean,
                remote_url=remote_url,
                repo_dir=repo_dir,
                commit_sha=commit_sha,
            )

        except Exception as e:
            logger.debug(f"Failed to discover git repository: {e}")
            return cls()

    def _get_remote_url(self) -> str:
        """Get the remote push URL.

        Returns the 'origin' remote push URL if it exists, otherwise returns
        the first remote alphabetically. Converts SSH/Git protocol URLs to HTTPS format.

        :return: The remote push URL in HTTPS format, or empty string if not found
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
                return self._normalize_url_to_https(url)

            # If origin doesn't exist, get all remotes
            result = subprocess.run(
                ["git", "remote"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                remotes = result.stdout.strip().split("\n")
                if remotes:
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
                        return self._normalize_url_to_https(url)

            return ""

        except Exception:
            return ""

    def _normalize_url_to_https(self, url: str) -> str:
        """Convert SSH or Git protocol URLs to HTTPS format.

        Examples:
            git@github.com:user/repo.git -> https://github.com/user/repo
            https://github.com/user/repo.git -> https://github.com/user/repo

        :param url: The Git URL to normalize
        :return: The normalized HTTPS URL
        """
        # Remove .git suffix first
        url = url.removesuffix(".git")

        # Handle SSH format: git@host:path or user@host:path
        if url.startswith("git@"):
            parts = url.split("@", 1)
            if len(parts) == 2:
                host_and_path = parts[1].replace(":", "/", 1)
                return f"https://{host_and_path}"

        return url

    def _get_remote_host(self, url: str) -> str:
        """Get the remote host name from a normalized HTTPS URL.

        :param url: URL that has been normalized to HTTPS format by _normalize_url_to_https
        :return: The host name (e.g., "github.com", "gitlab.com")
        """
        parts = url.split("//", 1)
        if len(parts) < 2:
            return ""

        # Get everything after "//" and split by "/"
        host_and_path = parts[1]
        parts = host_and_path.split("/", 1)
        if len(parts) < 2:
            return ""
        host = host_and_path.split("/")[0]

        return host

    def _get_file_path(self, path: Path | str) -> str:
        """Get the path relative to the repository root directory.

        :param path: Absolute or relative path to a file
        :return: Path relative to repo_dir as string, or empty string if failed
        """
        try:
            path_obj = Path(path).resolve()
            relative_path = path_obj.relative_to(self.repo_dir)
            return str(relative_path)
        except Exception as e:
            logger.warning(f"Failed to get relative path for {path}: {e}")
            return ""

    def build_url(self, path: Path | str, line_number: int) -> str:
        """Build a git URL for the given path.

        :param path: Path to a file
        :param line_number: Line number of the code file
        :return: Path relative to repo_dir
        """
        if not self.is_valid:
            logger.warning("GitConfig is not valid, cannot build URL")
            return ""
        host_name = self._get_remote_host(self.remote_url)
        git_file_path = self._get_file_path(path)
        if not host_name:
            logger.warning(f"Failed to extract host name from remote URL: {self.remote_url}")
            return ""
        if not git_file_path:
            return ""
        builder = GIT_URL_BUILDER_REGISTRY.get(host_name)
        if not builder:
            logger.warning(f"URL builder for {host_name} is not implemented")
            return ""
        url = builder.build_url(self.remote_url, git_file_path, self.commit_sha, line_number, self.is_tree_clean)
        return url


def config_from_root(path: pathlib.Path | str = ".flyte/config.yaml") -> flyte.config.Config | None:
    """Get the config file from the git root directory.

    By default, the config file is expected to be in `.flyte/config.yaml` in the git root directory.

    :param path: Path to the config file relative to git root directory (default: ".flyte/config.yaml")
    :return: Config object if found, None otherwise
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
