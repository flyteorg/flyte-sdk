import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flyte.git._config import GitStatus


class TestGitStatus:

    def test_get_remote_url_with_origin_https(self):
        git_status = GitStatus()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/user/repo.git\n"

        with patch("subprocess.run", return_value=mock_result):
            result = git_status._get_remote_url()

            # HTTPS URL should have .git suffix removed
            assert result == "https://github.com/user/repo"


    def test_get_remote_url_exception_handling(self):
        git_status = GitStatus()

        with patch("subprocess.run", side_effect=Exception("Git not found")):
            result = git_status._get_remote_url()

            # Should return empty string on exception
            assert result == ""

    def test_normalize_url_to_https_ssh_format(self):
        git_status = GitStatus()

        # SSH format: git@github.com:user/repo.git
        ssh_url = "git@github.com:user/repo.git"
        result = git_status._normalize_url_to_https(ssh_url)

        assert result == "https://github.com/user/repo"

    def test_normalize_url_to_https_https_format(self):
        git_status = GitStatus()

        https_url = "https://github.com/user/repo.git"
        result = git_status._normalize_url_to_https(https_url)

        assert result == "https://github.com/user/repo"

    def test_get_remote_host(self):
        git_status = GitStatus()

        url = "https://github.com/user/repo"
        result = git_status._get_remote_host(url)

        assert result == "github.com"

    def test_get_remote_host_no_protocol(self):
        git_status = GitStatus()

        url = "github.com/user/repo"
        result = git_status._get_remote_host(url)

        assert result == ""

    def test_get_remote_host_no_path(self):
        git_status = GitStatus()

        url = "https://github.com"
        result = git_status._get_remote_host(url)

        assert result == ""

    def test_get_file_path_correct_path(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(repo_dir=repo_dir)

        file_path = Path("/home/user/project/src/main.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status._get_file_path(file_path)

        assert result == "src/main.py"

    def test_get_file_path_incorrect_path(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(repo_dir=repo_dir)

        file_path = Path("/home/other/file.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status._get_file_path(file_path)

        assert result == ""

    def test_build_url_github_clean_tree(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            is_tree_clean=True,
            remote_url="https://github.com/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = Path("/home/user/project/src/main.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status.build_url(file_path, line_number=42)

        assert result == "https://github.com/user/repo/blob/abc123/src/main.py#L42"

    def test_build_url_github_dirty_tree(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            is_tree_clean=False,
            remote_url="https://github.com/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = Path("/home/user/project/src/main.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status.build_url(file_path, line_number=42)

        assert result == "https://github.com/user/repo/blob/abc123/src/main.py"

    def test_build_url_gitlab(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            is_tree_clean=True,
            remote_url="https://gitlab.com/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = Path("/home/user/project/src/main.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status.build_url(file_path, line_number=42)

        assert result == "https://gitlab.com/user/repo/-/blob/abc123/src/main.py#L42"

    def test_build_url_invalid_git_status(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=False,
            remote_url="https://github.com/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = repo_dir / "src" / "main.py"
        result = git_status.build_url(file_path, line_number=42)

        assert result == ""

    def test_build_url_invalid_remote_url(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            remote_url="invalid-url",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = repo_dir / "src" / "main.py"
        result = git_status.build_url(file_path, line_number=42)

        assert result == ""

    def test_build_url_file_outside_repo(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            remote_url="https://github.com/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = Path("/home/other/file.py")

        with patch.object(Path, 'resolve', side_effect=lambda: file_path):
            result = git_status.build_url(file_path, line_number=42)

        assert result == ""

    def test_build_url_unsupported_host(self):
        repo_dir = Path("/home/user/project")
        git_status = GitStatus(
            is_valid=True,
            remote_url="https://bitbucket.org/user/repo",
            repo_dir=repo_dir,
            commit_sha="abc123"
        )

        file_path = repo_dir / "src" / "main.py"
        result = git_status.build_url(file_path, line_number=42)

        assert result == ""

