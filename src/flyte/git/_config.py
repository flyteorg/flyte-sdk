import pathlib
import subprocess
from pathlib import Path
from typing import Dict, Protocol

import httpx

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

class GitConfig:
    """Configuration and information about the current Git repository."""

    is_valid: bool
    is_tree_clean: bool
    remote_url: str
    repo_dir: Path
    commit_sha: str
    branch_name: str

    def __init__(self):
        """Initialize all Git-related variables using Git commands.

        If Git is not installed or .git does not exist, marks is_git_repo as False and returns.
        """
        self.is_valid = False
        self.is_tree_clean = False
        self.remote_url = ""
        self.repo_dir = Path.cwd()
        self.commit_sha = ""
        self.branch_name = ""

        try:
            # Check if we're in a git repository and get the root directory
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
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
                return

            # Get remote URL
            self.remote_url = self._get_remote_url()
            if not self.remote_url:
                return
            self.is_valid = True

        except Exception as e:
            self.is_valid = False

    def _get_remote_url(self) -> str:
        """Get the remote push URL.

        Returns the 'origin' remote push URL if it exists, otherwise returns
        the first remote alphabetically. Converts SSH/Git protocol URLs to HTTPS format.
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
                        return self._normalize_url_to_https(url)

            return ""

        except Exception:
            return ""

    def _normalize_url_to_https(self, url: str) -> str:
        """Convert SSH or Git protocol URLs to HTTPS format.

        Examples:
            git@github.com:user/repo.git -> https://github.com/user/repo
            https://github.com/user/repo.git -> https://github.com/user/repo
        """
        # Remove .git suffix first
        if url.endswith(".git"):
            url = url[:-4]

        # Handle SSH format: git@host:path or user@host:path
        if url.startswith("git@"):
            parts = url.split("@", 1)
            if len(parts) == 2:
                host_and_path = parts[1].replace(":", "/", 1)
                return f"https://{host_and_path}"

        return url

    def _get_remote_host(self, url: str) -> str:
        """Get the remote host name from a normalized HTTPS URL.

        Args:
            url: URL that has been normalized to HTTPS format by _normalize_url_to_https

        Returns:
            The host name (e.g., "github.com", "gitlab.com")
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
            return ""

    def build_url(self, path: Path | str, line_number: int) -> str:
        """Build a git URL for the given path."""
        if not self.is_valid:
            return ""
        host_name = self._get_remote_host(self.remote_url)
        git_file_path = self.get_file_path(path)
        if not host_name or not git_file_path:
            return ""
        builder = GIT_URL_BUILDER_REGISTRY.get(host_name)
        if not builder:
            return ""
        url = builder.build_url(self.remote_url, git_file_path, self.commit_sha, line_number, self.is_tree_clean)
        return url

    @staticmethod
    async def is_valid_url(url: str) -> bool:
        """Validate a git URL by sending an HTTP request.

        Args:
            url: The URL to validate

        Returns:
            True if the URL returns a success response, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=10.0)
                if not response.is_success:
                    return False
                return True
        except Exception as e:
            return False


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
