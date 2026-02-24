"""Tests for agent SDK utility functions in flyteplugins.codegen.execution.agent_sdk."""

import pytest

from flyteplugins.codegen.execution.agent_sdk import _classify_bash_command


class TestClassifyBashCommand:
    def test_pytest_command(self):
        assert _classify_bash_command("pytest tests.py") == "pytest"

    def test_pytest_with_flags(self):
        assert _classify_bash_command("pytest tests.py -v --tb=short") == "pytest"

    def test_python_m_pytest(self):
        """'python -m pytest' has 'pytest' in tokens."""
        assert _classify_bash_command("python -m pytest tests.py") == "pytest"

    def test_ls_command(self):
        assert _classify_bash_command("ls -la") == "allow"

    def test_cat_command(self):
        assert _classify_bash_command("cat file.txt") == "allow"

    def test_mkdir_command(self):
        assert _classify_bash_command("mkdir -p /tmp/test") == "allow"

    def test_curl_denied(self):
        assert _classify_bash_command("curl https://example.com") == "deny"

    def test_apt_get_denied(self):
        assert _classify_bash_command("apt-get install gcc") == "deny"

    def test_pip_denied(self):
        assert _classify_bash_command("pip install numpy") == "deny"

    def test_wget_denied(self):
        assert _classify_bash_command("wget https://example.com/file") == "deny"

    def test_empty_string(self):
        assert _classify_bash_command("") == "deny"

    def test_whitespace_only(self):
        assert _classify_bash_command("   ") == "deny"

    def test_safe_prefixes(self):
        """Test all defined safe prefixes."""
        safe_cmds = ["ls", "pwd", "cat", "head", "tail", "grep", "wc", "mkdir", "touch", "rm", "mv", "cp", "echo",
                     "sed", "awk", "find"]
        for cmd in safe_cmds:
            assert _classify_bash_command(cmd) == "allow", f"{cmd} should be allowed"

    def test_invalid_shell_syntax(self):
        """Malformed shell syntax should be denied."""
        assert _classify_bash_command("'unclosed quote") == "deny"
