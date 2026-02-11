from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from flyte.cli.main import main


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


def test_create_secret_no_value(runner: CliRunner):
    result = runner.invoke(main, ["create", "secret", "my_secret"])
    assert result.exit_code == 1
    assert result.stdout.startswith("Enter secret value: ")


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_secret_value(mock_cli_config, mock_secret_create, runner: CliRunner):
    mock_secret_create.return_value = None

    secret_value = "my_value"

    result = runner.invoke(main, ["create", "secret", "my_secret", "--value", secret_value])
    assert result.exit_code == 0, result.stderr
    mock_secret_create.assert_called_once_with(name="my_secret", value=b"my_value", type="regular")


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_secret_from_file(mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path):
    mock_secret_create.return_value = None

    secret_value = "my_value"
    with open(tmp_path / "secret.txt", "w") as f:
        f.write(secret_value)

    result = runner.invoke(main, ["create", "secret", "my_secret", "--from-file", str(tmp_path / "secret.txt")])
    assert result.exit_code == 0, result.stderr
    mock_secret_create.assert_called_once_with(name="my_secret", value=b"my_value", type="regular")


def test_create_secret_invalid_combination(runner: CliRunner):
    result = runner.invoke(main, ["create", "secret", "my_secret", "--value", "my_value", "--from-file", "my_file"])
    assert result.exit_code == 2
    # The error message includes all mutually exclusive options
    assert "Illegal usage" in result.stderr
    assert "are mutually exclusive" in result.stderr


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_interactive(mock_cli_config, mock_secret_create, runner: CliRunner):
    """Test creating image pull secret with interactive prompts."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        ["create", "secret", "my_secret", "--type", "image_pull"],
        input="ghcr.io\nmyuser\nmytoken\n",
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value is valid dockerconfigjson
    import json

    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_explicit_credentials(mock_cli_config, mock_secret_create, runner: CliRunner):
    """Test creating image pull secret with explicit credentials."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "docker.io",
            "--username",
            "testuser",
            "--password",
            "testpass",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value is valid dockerconfigjson
    import base64
    import json

    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "docker.io" in config["auths"]

    # Verify credentials
    auth_token = config["auths"]["docker.io"]["auth"]
    decoded = base64.b64decode(auth_token).decode()
    assert decoded == "testuser:testpass"


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_explicit_credentials_prompt_password(
    mock_cli_config, mock_secret_create, runner: CliRunner
):
    """Test creating image pull secret with registry and username, prompting for password."""
    mock_secret_create.return_value = None

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "ghcr.io",
            "--username",
            "user",
        ],
        input="mypassword\n",
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_from_docker_config(mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path):
    """Test creating image pull secret from Docker config file."""
    mock_secret_create.return_value = None

    # Create a test Docker config
    import json

    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--from-docker-config",
            "--docker-config-path",
            str(config_file),
            "--registries",
            "ghcr.io",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert mock_secret_create.called
    call_args = mock_secret_create.call_args
    assert call_args[1]["name"] == "my_secret"
    assert call_args[1]["type"] == "image_pull"

    # Verify the value contains only the specified registry
    value = call_args[1]["value"]
    config = json.loads(value)
    assert "auths" in config
    assert "ghcr.io" in config["auths"]
    assert "docker.io" not in config["auths"]


@patch("flyte.remote.Secret.create")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_create_image_pull_secret_from_docker_config_all_registries(
    mock_cli_config, mock_secret_create, runner: CliRunner, tmp_path
):
    """Test creating image pull secret from Docker config with all registries."""
    mock_secret_create.return_value = None

    import json

    config_file = tmp_path / "config.json"
    test_config = {
        "auths": {
            "docker.io": {"auth": "dGVzdDp0ZXN0"},
            "ghcr.io": {"auth": "dXNlcjpwYXNz"},
        }
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f)

    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--from-docker-config",
            "--docker-config-path",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.stderr
    call_args = mock_secret_create.call_args
    value = call_args[1]["value"]
    config = json.loads(value)
    assert "docker.io" in config["auths"]
    assert "ghcr.io" in config["auths"]


def test_create_image_pull_secret_invalid_combination_registry_and_from_docker_config(runner: CliRunner):
    """Test that --registry and --from-docker-config are mutually exclusive."""
    result = runner.invoke(
        main,
        [
            "create",
            "secret",
            "my_secret",
            "--type",
            "image_pull",
            "--registry",
            "ghcr.io",
            "--from-docker-config",
        ],
    )
    assert result.exit_code == 2
    error_msg = "are mutually exclusive"
    assert error_msg in result.stderr


def test_config_with_params_preserves_local():
    """Verify Config.with_params() doesn't drop LocalConfig."""
    from flyte.config._config import Config, LocalConfig, PlatformConfig, TaskConfig

    cfg = Config(local=LocalConfig(persistence=True), source=None)
    updated = cfg.with_params(PlatformConfig(), TaskConfig())
    assert updated.local.persistence is True


def test_create_config_with_local_persistence(runner: CliRunner, tmp_path):
    """Test that --local-persistence writes the local.persistence field to the config YAML."""
    outpath = str(tmp_path / "config.yaml")
    result = runner.invoke(
        main,
        ["create", "config", "--endpoint", "dns:///test.example.com", "--local-persistence", "-o", outpath, "--force"],
    )
    assert result.exit_code == 0, result.output
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert "local" in d
    assert d["local"]["persistence"] is True


def test_create_config_without_local_persistence(runner: CliRunner, tmp_path):
    """Test that without --local-persistence the local section is omitted."""
    outpath = str(tmp_path / "config.yaml")
    result = runner.invoke(
        main,
        ["create", "config", "--endpoint", "dns:///test.example.com", "-o", outpath, "--force"],
    )
    assert result.exit_code == 0, result.output
    with open(outpath) as f:
        d = yaml.safe_load(f)
    assert d.get("local") is None
