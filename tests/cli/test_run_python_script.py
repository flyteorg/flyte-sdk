"""Tests for `flyte run python-script`, focused on the `--plugin-config` flag."""

import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._run import run

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TEST_CODE_PATH = pathlib.Path(__file__).parent
RUN_TESTDATA = TEST_CODE_PATH / "run_testdata"
HELLO_WORLD_PY = RUN_TESTDATA / "hello_world.py"


@pytest.fixture
def runner():
    return CliRunner()


def _normalized(output: str) -> str:
    """Collapse rich_click's word-wrapped, ANSI-colored error box into a plain single-spaced line."""
    plain = _ANSI_RE.sub("", output)
    plain = re.sub(r"[│╭╮╰╯─]", " ", plain)
    return " ".join(plain.split())


@pytest.fixture
def mock_run_python_script():
    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "http://example.com/run/test-run"
    with patch("flyte._run_python_script.run_python_script", return_value=mock_run) as m:
        yield m


def test_plugin_config_flag_builds_instance(tmp_path, runner, mock_run_python_script):
    from tests.cli.run_testdata.dummy_plugin_config import DummyNodeConfig, DummyPluginConfig

    plugin_yaml = tmp_path / "plugin.yaml"
    plugin_yaml.write_text(
        "plugin: tests.cli.run_testdata.dummy_plugin_config.DummyPluginConfig\n"
        "config:\n"
        "  nodes:\n"
        "    - group_name: workers\n"
        "      replicas: 2\n"
        "  enabled: true\n"
    )

    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--plugin-config",
            str(plugin_yaml),
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_run_python_script.call_args.kwargs
    assert kwargs["plugin_config"] == DummyPluginConfig(
        nodes=[DummyNodeConfig(group_name="workers", replicas=2)], enabled=True
    )


def test_plugin_config_missing_plugin_key_errors(tmp_path, runner, mock_run_python_script):
    plugin_yaml = tmp_path / "plugin.yaml"
    plugin_yaml.write_text("config:\n  a: 1\n")

    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--plugin-config",
            str(plugin_yaml),
        ],
    )

    assert result.exit_code != 0
    assert "top-level 'plugin' key" in result.output
    mock_run_python_script.assert_not_called()


def test_plugin_config_unresolvable_class_errors(tmp_path, runner, mock_run_python_script):
    plugin_yaml = tmp_path / "plugin.yaml"
    plugin_yaml.write_text("plugin: nonexistent.module.ClassName\n")

    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--plugin-config",
            str(plugin_yaml),
        ],
    )

    assert result.exit_code != 0
    assert "Could not load plugin config" in result.output
    mock_run_python_script.assert_not_called()


def test_no_plugin_config_by_default(runner, mock_run_python_script):
    result = runner.invoke(
        run,
        ["--project", "p", "--domain", "d", "python-script", str(HELLO_WORLD_PY)],
    )
    assert result.exit_code == 0, result.output
    assert mock_run_python_script.call_args.kwargs["plugin_config"] is None


# ---------------------------------------------------------------------------
# --clustered
# ---------------------------------------------------------------------------


def test_clustered_requires_replicas_and_nproc_per_node(runner, mock_run_python_script):
    result = runner.invoke(
        run,
        ["--project", "p", "--domain", "d", "python-script", str(HELLO_WORLD_PY), "--clustered"],
    )
    assert result.exit_code != 0
    assert "requires both --replicas and --nproc-per-node" in _normalized(result.output)
    mock_run_python_script.assert_not_called()


def test_clustered_and_plugin_config_mutually_exclusive(tmp_path, runner, mock_run_python_script):
    plugin_yaml = tmp_path / "plugin.yaml"
    plugin_yaml.write_text("plugin: tests.cli.run_testdata.dummy_plugin_config.DummyPluginConfig\nconfig: {}\n")

    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--clustered",
            "--replicas",
            "2",
            "--nproc-per-node",
            "1",
            "--plugin-config",
            str(plugin_yaml),
        ],
    )
    assert result.exit_code != 0
    assert "are mutually exclusive" in _normalized(result.output)
    mock_run_python_script.assert_not_called()


def test_replicas_without_clustered_errors(runner, mock_run_python_script):
    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--replicas",
            "2",
            "--nproc-per-node",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "require --clustered" in _normalized(result.output)
    mock_run_python_script.assert_not_called()


def test_clustered_flags_forwarded(runner, mock_run_python_script):
    from flyte.clustered import ClusterFailurePolicy, TorchRun

    result = runner.invoke(
        run,
        [
            "--project",
            "p",
            "--domain",
            "d",
            "python-script",
            str(HELLO_WORLD_PY),
            "--clustered",
            "--replicas",
            "4",
            "--nproc-per-node",
            "8",
            "--rdzv-backend",
            "c10d",
            "--torchrun-max-restarts",
            "2",
            "--cluster-max-restarts",
            "3",
            "--restart-on-host-maintenance",
            "--ttl-seconds-after-finished",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_run_python_script.call_args.kwargs
    assert kwargs["clustered"] is True
    assert kwargs["replicas"] == 4
    assert kwargs["nproc_per_node"] == 8
    assert kwargs["runtime"] == TorchRun(rdzv_backend="c10d", max_restarts=2)
    assert kwargs["failure_policy"] == ClusterFailurePolicy(max_restarts=3, restart_on_host_maintenance=True)
    assert kwargs["ttl_seconds_after_finished"] == 300


def test_no_clustered_kwargs_by_default(runner, mock_run_python_script):
    result = runner.invoke(
        run,
        ["--project", "p", "--domain", "d", "python-script", str(HELLO_WORLD_PY)],
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_run_python_script.call_args.kwargs
    assert kwargs["clustered"] is False
    assert kwargs["replicas"] is None
    assert kwargs["nproc_per_node"] is None
    assert kwargs["runtime"] is None
    assert kwargs["failure_policy"] is None
    assert kwargs["ttl_seconds_after_finished"] is None
