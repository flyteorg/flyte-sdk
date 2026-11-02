"""
Tests for how the `flyte-mcp` command picks the Flyte config and endpoint it serves against.

Without `--config` the server uses the CLI's search order, which prefers a config in the
working directory or its git repo over FLYTECTL_CONFIG. An MCP client usually launches the
server from inside a project, so these tests check that the server can be pinned with
`--config` / `--endpoint` (or their env vars), and that it always says on stderr which
config and endpoint it resolved, warning when the config came from the working directory.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import flyte
import flyte._initialize as init_mod
import flyte.remote
from flyte._bin import mcp as mcp_cli
from flyte.ai.mcp import FlyteMCPAppEnvironment

_ENV_VARS = (
    "FLYTE_MCP_CONFIG",
    "FLYTE_MCP_ENDPOINT",
    "FLYTECTL_CONFIG",
    "UCTL_CONFIG",
    "FLYTE_ADMIN_ENDPOINT",
)


def _write_config(path: pathlib.Path, endpoint: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"admin:\n  endpoint: {endpoint}\n")
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A project directory with its own .flyte/config.yaml and an empty home directory.

    The repo-local config points at one tenant; tests that pin the server point it at another.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = tmp_path / "repo"
    _write_config(repo / ".flyte" / "config.yaml", "dns:///repo-tenant.example.com")
    monkeypatch.chdir(repo)
    # Keep the search order from walking up into a real git repo around tmp_path.
    monkeypatch.setattr("flyte.config._reader._config_path_from_git_root", lambda: None)
    return SimpleNamespace(home=home, repo=repo, other=tmp_path / "other")


@pytest.fixture
def captured_init(monkeypatch):
    """Record the Config the server initializes Flyte with instead of connecting anywhere."""
    seen: dict = {}

    def _fake_init_from_config(cfg=None, **kwargs):
        seen["cfg"] = cfg

    monkeypatch.setattr(flyte, "init_from_config", _fake_init_from_config)
    monkeypatch.setattr(FlyteMCPAppEnvironment, "run_stdio", lambda self: seen.setdefault("env", self))
    return seen


def _invoke(*args: str, env: dict | None = None):
    # `--tools whoami` keeps the run off the network: no search corpus to fetch.
    return CliRunner().invoke(mcp_cli.main, ["--transport", "stdio", "--tools", "whoami", *args], env=env)


class TestDefaultSearch:
    def test_repo_local_config_is_used_and_flagged(self, workspace, captured_init):
        result = _invoke()
        assert result.exit_code == 0, result.output

        repo_cfg = (workspace.repo / ".flyte" / "config.yaml").resolve()
        assert captured_init["cfg"].platform.endpoint == "dns:///repo-tenant.example.com"
        assert (
            f"flyte-mcp: config {repo_cfg} (from default search order), "
            "endpoint dns:///repo-tenant.example.com (from config file)"
        ) in result.stderr
        assert "warning" in result.stderr
        assert "FLYTE_MCP_CONFIG" in result.stderr
        assert captured_init["env"].config_file == str(repo_cfg)

    def test_repo_local_config_still_beats_flytectl_config(self, workspace, captured_init, monkeypatch):
        # The CLI's search order is unchanged; only the MCP server gained a way to pin.
        other = _write_config(workspace.other / "config.yaml", "dns:///other-tenant.example.com")
        monkeypatch.setenv("FLYTECTL_CONFIG", str(other))

        result = _invoke()
        assert result.exit_code == 0, result.output
        assert captured_init["cfg"].platform.endpoint == "dns:///repo-tenant.example.com"

    def test_user_level_config_is_not_flagged(self, workspace, captured_init, monkeypatch):
        (workspace.repo / ".flyte" / "config.yaml").unlink()
        home_cfg = _write_config(workspace.home / ".flyte" / "config.yaml", "dns:///home-tenant.example.com")

        result = _invoke()
        assert result.exit_code == 0, result.output
        assert f"config {home_cfg.resolve()} (from default search order)" in result.stderr
        assert "warning" not in result.stderr

    def test_no_config_found_is_reported(self, workspace, captured_init):
        (workspace.repo / ".flyte" / "config.yaml").unlink()

        result = _invoke()
        assert result.exit_code == 0, result.output
        assert "flyte-mcp: config none (no config file found), endpoint none" in result.stderr
        assert captured_init["env"].config_file is None

    def test_admin_endpoint_env_var_is_named_as_the_source(self, workspace, captured_init, monkeypatch):
        monkeypatch.setenv("FLYTE_ADMIN_ENDPOINT", "dns:///env-tenant.example.com")

        result = _invoke()
        assert result.exit_code == 0, result.output
        assert "endpoint dns:///env-tenant.example.com (from FLYTE_ADMIN_ENDPOINT)" in result.stderr


class TestPinning:
    def test_config_flag_wins_over_repo_local_config(self, workspace, captured_init):
        other = _write_config(workspace.other / "config.yaml", "dns:///other-tenant.example.com")

        result = _invoke("--config", str(other))
        assert result.exit_code == 0, result.output
        assert captured_init["cfg"].platform.endpoint == "dns:///other-tenant.example.com"
        assert f"config {other.resolve()} (from --config)" in result.stderr
        assert "warning" not in result.stderr
        assert captured_init["env"].config_file == str(other.resolve())

    def test_config_env_var_wins_over_repo_local_config(self, workspace, captured_init):
        other = _write_config(workspace.other / "config.yaml", "dns:///other-tenant.example.com")

        result = _invoke(env={"FLYTE_MCP_CONFIG": str(other)})
        assert result.exit_code == 0, result.output
        assert captured_init["cfg"].platform.endpoint == "dns:///other-tenant.example.com"
        assert f"config {other.resolve()} (from FLYTE_MCP_CONFIG)" in result.stderr

    def test_endpoint_flag_overrides_the_config_endpoint(self, workspace, captured_init):
        result = _invoke("--endpoint", "dns:///pinned.example.com")
        assert result.exit_code == 0, result.output
        assert captured_init["cfg"].platform.endpoint == "dns:///pinned.example.com"
        assert "endpoint dns:///pinned.example.com (from --endpoint)" in result.stderr

    def test_endpoint_env_var_overrides_the_config_endpoint(self, workspace, captured_init):
        result = _invoke(env={"FLYTE_MCP_ENDPOINT": "dns:///pinned.example.com"})
        assert result.exit_code == 0, result.output
        assert captured_init["cfg"].platform.endpoint == "dns:///pinned.example.com"
        assert "endpoint dns:///pinned.example.com (from FLYTE_MCP_ENDPOINT)" in result.stderr

    def test_missing_config_file_is_an_error(self, workspace, captured_init):
        result = _invoke("--config", str(workspace.other / "missing.yaml"))
        assert result.exit_code != 0
        assert "cfg" not in captured_init

    def test_pinning_needs_init_from_config(self, workspace, captured_init):
        result = _invoke("--no-init-from-config", "--endpoint", "dns:///pinned.example.com")
        assert result.exit_code != 0
        assert "--no-init-from-config" in result.output
        assert "cfg" not in captured_init

    def test_the_startup_log_stays_off_stdout(self, workspace, captured_init):
        # Under stdio, stdout is the JSON-RPC channel.
        result = _invoke("--endpoint", "dns:///pinned.example.com")
        assert result.exit_code == 0, result.output
        assert "flyte-mcp: config" not in result.stdout


@pytest.mark.asyncio
async def test_whoami_reports_the_config_file(monkeypatch):
    cfg = init_mod._InitConfig(
        root_dir=pathlib.Path.cwd(),
        org="test-org",
        project="p",
        domain="d",
        client=SimpleNamespace(endpoint="dns:///pinned.example.com"),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(init_mod, "_init_config", cfg)

    class _FakeUserGet:
        async def aio(self):
            return SimpleNamespace(subject=lambda: "sub", name=lambda: "someone")

    monkeypatch.setattr(flyte.remote.User, "get", _FakeUserGet())

    env = FlyteMCPAppEnvironment(name="test-mcp", tools=["whoami"], config_file="/etc/flyte/config.yaml")
    whoami = env._mcp_server._tool_manager._tools["whoami"].fn
    out = await whoami()
    assert out["config_file"] == "/etc/flyte/config.yaml"
    assert out["endpoint"] == "dns:///pinned.example.com"
