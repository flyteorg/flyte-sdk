from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

import flyte
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte.cli._build import build


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_cli_config():
    cfg = Mock()
    cfg.output_format = "table-simple"
    cfg.log_level = None
    cfg.init.return_value = None
    return cfg


def test_build_force_flag_passed_to_build_images(runner, tmp_path, mock_cli_config):
    """--force is forwarded to flyte.build_images."""
    images_file = tmp_path / "images.py"
    # Variable name "my_env" is used as the subcommand, not the Environment's name attribute.
    images_file.write_text(
        "import flyte\n"
        "my_env = flyte.Environment(name='my_env',\n"
        "    image=flyte.Image.from_base('python:3.12').clone(\n"
        "        registry='reg', name='img'))\n"
    )

    mock_cache = ImageCache(image_lookup={"my_env": "reg/img:abc123"})

    with patch("flyte.build_images", return_value=mock_cache) as mock_build:
        result = runner.invoke(
            build,
            ["--force", str(images_file), "my_env"],
            obj=mock_cli_config,
        )

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs.get("force") is True


def test_build_force_defaults_to_false(runner, tmp_path, mock_cli_config):
    """force defaults to False when --force is not passed."""
    images_file = tmp_path / "images.py"
    # Variable name "my_env" is used as the subcommand, not the Environment's name attribute.
    images_file.write_text(
        "import flyte\n"
        "my_env = flyte.Environment(name='my_env',\n"
        "    image=flyte.Image.from_base('python:3.12').clone(\n"
        "        registry='reg', name='img'))\n"
    )

    mock_cache = ImageCache(image_lookup={"my_env": "reg/img:abc123"})

    with patch("flyte.build_images", return_value=mock_cache) as mock_build:
        result = runner.invoke(
            build,
            [str(images_file), "my_env"],
            obj=mock_cli_config,
        )

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args[1]
    assert not call_kwargs.get("force", False)
