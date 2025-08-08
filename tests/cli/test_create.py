import pytest
import click
from io import BytesIO
from click.testing import CliRunner
from unittest.mock import patch, Mock
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
    mock_secret_create.assert_called_once_with(name="my_secret", value="my_value", type="regular")


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
    error_msg = "Illegal usage: options 'value' and 'from_file' are mutually exclusive"
    assert error_msg in result.stderr
