"""Tests for the ``flyte-mcp`` CLI's central-mode flags.

``_build_env`` is stubbed out, so nothing initializes Flyte, clones a corpus, or binds a port.
"""

import pytest
from click.testing import CliRunner

from flyte._bin import mcp as mcp_cli

#: Central mode requires a Host allowlist, so tests that only care about other flags pass one.
HOSTS_FLAG = ["--allowed-hosts", "mcp.union.ai"]


@pytest.fixture(autouse=True)
def _no_env_host_allowlist(monkeypatch):
    """Keep the ambient environment from satisfying (or breaking) the --central host check."""
    monkeypatch.delenv("FLYTE_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("FLYTE_MCP_ALLOWED_ORIGINS", raising=False)


@pytest.fixture
def captured(monkeypatch):
    """Replace ``_build_env``/``_serve_http`` and return the kwargs the CLI passed."""
    seen: dict = {}

    def fake_build_env(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(mcp_cli, "_build_env", fake_build_env)
    monkeypatch.setattr(mcp_cli, "_serve_http", lambda env, mount: None)
    return seen


def _run(args):
    return CliRunner().invoke(mcp_cli.main, args)


class TestCentralFlag:
    def test_default_is_off(self, captured, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)
        result = _run(["--no-init-from-config"])
        assert result.exit_code == 0, result.output
        assert captured["central_mode"] is False

    def test_flag_enables_central_mode(self, captured, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)
        result = _run(["--central", *HOSTS_FLAG])
        assert result.exit_code == 0, result.output
        assert captured["central_mode"] is True
        # Central mode has no single tenant to bind, so config init is skipped.
        assert captured["init_from_config"] is False

    @pytest.mark.parametrize("value", ["1", "true", "True"])
    def test_env_var_enables_central_mode(self, captured, monkeypatch, value):
        monkeypatch.setenv("FLYTE_MCP_CENTRAL", value)
        result = _run(["--no-init-from-config", *HOSTS_FLAG])
        assert result.exit_code == 0, result.output
        assert captured["central_mode"] is True

    @pytest.mark.parametrize("value", ["0", "false", ""])
    def test_env_var_falsy_keeps_central_mode_off(self, captured, monkeypatch, value):
        monkeypatch.setenv("FLYTE_MCP_CENTRAL", value)
        result = _run(["--no-init-from-config"])
        assert result.exit_code == 0, result.output
        assert captured["central_mode"] is False

    def test_central_rejected_with_stdio(self, captured, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)
        result = _run(["--central", "--transport", "stdio"])
        assert result.exit_code != 0
        assert "--central requires an HTTP transport" in result.output

    def test_allowlist_flags_are_parsed(self, captured, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)
        result = _run(
            [
                "--central",
                "--allowed-endpoint-suffixes",
                ".a.example.com, .b.example.com",
                "--allowed-hosts",
                "mcp.union.ai,mcp.union.ai:443",
            ]
        )
        assert result.exit_code == 0, result.output
        assert captured["allowed_endpoint_suffixes"] == [".a.example.com", ".b.example.com"]
        assert captured["allowed_hosts"] == ["mcp.union.ai", "mcp.union.ai:443"]

    def test_allowlist_flags_default_to_none(self, captured, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)
        result = _run(["--no-init-from-config"])
        assert result.exit_code == 0, result.output
        assert captured["allowed_endpoint_suffixes"] is None
        assert captured["allowed_hosts"] is None


class TestCentralRequiresHostAllowlist:
    """Serving centrally without a Host allowlist leaves DNS-rebinding protection off."""

    @pytest.fixture(autouse=True)
    def _central_off_by_default(self, monkeypatch):
        monkeypatch.delenv("FLYTE_MCP_CENTRAL", raising=False)

    def test_central_without_hosts_is_rejected(self, captured):
        result = _run(["--central"])
        assert result.exit_code != 0
        assert "--allowed-hosts" in result.output
        assert "FLYTE_MCP_ALLOWED_HOSTS" in result.output
        # The failure happens before anything is built.
        assert captured == {}

    def test_env_central_without_hosts_is_rejected(self, captured, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_CENTRAL", "1")
        result = _run(["--no-init-from-config"])
        assert result.exit_code != 0
        assert "--allowed-hosts" in result.output

    def test_flag_satisfies_the_requirement(self, captured):
        result = _run(["--central", "--allowed-hosts", "mcp.union.ai,mcp.union.ai:443"])
        assert result.exit_code == 0, result.output
        assert captured["allowed_hosts"] == ["mcp.union.ai", "mcp.union.ai:443"]

    def test_env_var_satisfies_the_requirement(self, captured, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_ALLOWED_HOSTS", "mcp.union.ai")
        result = _run(["--central"])
        assert result.exit_code == 0, result.output
        # The env var reaches the environment through resolved_allowed_hosts, not this flag.
        assert captured["allowed_hosts"] is None

    def test_origins_env_var_also_satisfies_it(self, captured, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_ALLOWED_ORIGINS", "https://mcp.union.ai")
        result = _run(["--central"])
        assert result.exit_code == 0, result.output

    def test_non_central_needs_no_allowlist(self, captured):
        result = _run(["--no-init-from-config"])
        assert result.exit_code == 0, result.output
